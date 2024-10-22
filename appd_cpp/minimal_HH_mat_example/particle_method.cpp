#include "particle_method.h"
#include <boost/numeric/odeint.hpp>
#ifndef MATLAB_DATAIO
#define H5_BUILT_AS_DYNAMIC_LIB
#include <H5Cpp.h>
#endif
#include "runge_kutta_bogacki_shampine32.h"
//A lower order method is implemented and used since the HH neurons are a bit stiff,
//and breaks step control of ode45. A lower order ode23 is more suitable. 
#define MATH_ERRNO 1
#define MATH_ERREXCEPT 2
//#define AVG_SPEED_LEVELSET
#include <cmath>
#include <list>
#include <vector>
#include<array>
#include <algorithm>
#include <unordered_map>
#include <new>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <chrono>
#include <stdexcept>
//#define DEBUG//Comment this to supress debug outputs.
#ifdef DEBUG
#define ADAPTIVE_SPLIT_NO_PARALLEL
#define DISPLAY_COMBINE
#endif // DEBUG
//#define ADAPTIVE_SPLIT_NO_PARALLEL
#ifdef DEBUG_DEADLOCK
#include <mutex>
#include <thread>
#include <condition_variable>
#define ADAPTIVE_SPLIT_NO_PARALLEL
#endif

struct Reduce_sum {
    //Required for doing parallel reduced sum. 
    value_type value;
    Reduce_sum() : value(0.0) {}
    Reduce_sum(Reduce_sum& s, tbb::split) : value(0.0) {}
    void operator()(const tbb::blocked_range<std::vector<value_type>::iterator>& r) {
        value = std::accumulate(r.begin(), r.end(), value);
    }
    void join(Reduce_sum& rhs) { value += rhs.value; }
};
struct Particle_reduce_sum {
    //Required for doing parallel reduced sum. 
    value_type sum;
    Particle_reduce_sum() : sum(0.0) {}
    Particle_reduce_sum(Particle_reduce_sum& s, tbb::split) : sum(0.0) {}
    void operator()(const tbb::blocked_range<tbb::concurrent_vector<Particle>::iterator>& r) {
        for (auto itr = r.begin(); itr < r.end(); itr++) {
            sum += itr->weight;
        }
    }
    void join(Particle_reduce_sum& rhs) { sum += rhs.sum; }
};
struct Particle_reduce_max {
    //Required for doing parallel reduced sum. 
    value_type maxval;
    Particle_reduce_max() : maxval(0.0) {}
    Particle_reduce_max(Particle_reduce_max& s, tbb::split) : maxval(0.0) {}
    void operator()(const tbb::blocked_range<tbb::concurrent_vector<Particle>::iterator>& r) {
        for (auto itr = r.begin(); itr < r.end(); itr++) {
            maxval = itr->weight > maxval ? itr->weight : maxval;
        }
    }
    void join(Particle_reduce_max& rhs) { maxval = maxval > rhs.maxval ? maxval : rhs.maxval; }
};

value_type Particle::density_at(const State_variable& location) const {
    const State_variable difference = location - center_location;
    const value_type equiv_norm_squared = difference.transpose() * covariance_matrix.llt().solve(difference);
    //weight * sqrt(determinant(M)) / (2 pi)^(dim/2) * exp(-equiv_norm_squared/(2)), where equiv_distance = -x^T M^(-1) x.
    //M is the covariance_matrix_matrix
    const value_type rval = weight / covariance_matrix.llt().matrixL().determinant() / pow(2.0*pi, difference.size() / 2.0) * exp(-equiv_norm_squared / (2.0));
    return rval;
}
Center_level_set Particle::to_center_level_set() const{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> EigenSolver;
    EigenSolver.compute(covariance_matrix);
    //covariance_sqrt can be any "square root" of the covariance matrix. We start from an orthogonal one. 
    const Matrix_type covariance_sqrt = (Eigen::VectorXd::Ones(center_location.size()) * EigenSolver.eigenvalues().transpose().cwiseSqrt()).array() *
        EigenSolver.eigenvectors().array();
    Center_level_set rval(this->center_location, covariance_sqrt);
    return rval;
}
value_type Particle::density_projection_at_coordinate(const State_variable& location, const std::vector<bool>& range_dimensions) const {
    const int dim_of_range = std::count(range_dimensions.begin(), range_dimensions.end(), true);
    Matrix_type sub_covariance_matrix(dim_of_range, dim_of_range);
    const State_variable difference = location - center_location;
    State_variable sub_difference(dim_of_range);
    const int space_dim = location.size();
    int sub_dim_index = 0;
    for (int i = 0; i < space_dim; i++) {
        if (range_dimensions[i])
            sub_difference[sub_dim_index++] = difference[i];
    }
    int sub_row_dim_idx = 0;
    for (int row_idx = 0; row_idx < space_dim; row_idx++) {
        if (range_dimensions[row_idx]) {
            int sub_col_dim_idx = 0;
            for (int col_idx = 0; col_idx < space_dim; col_idx++) {
                if (range_dimensions[col_idx])
                    sub_covariance_matrix(sub_row_dim_idx, sub_col_dim_idx++) = covariance_matrix(row_idx, col_idx);
            }
            sub_row_dim_idx++;
        }
    }
    const value_type equiv_norm_squared = sub_difference.transpose() * sub_covariance_matrix.llt().solve(sub_difference);
    const value_type rval = weight / sub_covariance_matrix.llt().matrixL().determinant() / pow(4.0*pi, dim_of_range / 2.0) * exp(-equiv_norm_squared / (4.0));
    return rval;
}
std::array<Particle, 3> split_particle_in_direction(const Particle x, const State_variable direction, const value_type singular_value);
value_type Population_density::density_at(const State_variable& location) const {
    if (dimension != location.size()) {
        std::cout << "Dimension do not match" << std::endl;
        return NAN;
    }
    value_type sum = 0.0;
    for (Particle x : p_vect) {
        sum += x.density_at(location);
    }
    return sum;
}value_type Population_density::density_projection_at_coordinate(const State_variable& location, const std::vector<bool>& range_dimensions) const {
    if (dimension != location.size()) {
        std::cout << "Dimension do not match" << std::endl;
        return NAN;
    }
    value_type sum = 0.0;
    for (Particle x : p_vect) {
        sum += x.density_projection_at_coordinate(location, range_dimensions);
    }
    return sum;
}
std::vector<value_type> Population_density::density_at(const std::vector<State_variable>& location) const {
    std::vector<value_type> rval(location.size());
    tbb::parallel_for(tbb::blocked_range<int>(0, location.size()), [&](tbb::blocked_range<int>&) {
        for (int i = 0; i < location.size(); i++) {
            rval[i] = density_at(location[i]);
        }
    });
    return rval;
}
std::vector<value_type> Population_density::density_projection_at_coordinate(const std::vector<State_variable>& location, const std::vector<bool>& range_dimensions) const {
    std::vector<value_type> rval(location.size());
    tbb::parallel_for(tbb::blocked_range<int>(0, location.size()), [&](tbb::blocked_range<int>&) {
        for (int i = 0; i < location.size(); i++) {
            rval[i] = density_projection_at_coordinate(location[i], range_dimensions);
        }
    });

    return rval;
}
Center_level_set abs(const Center_level_set& p1) {
    return Center_level_set(p1.center.cwiseAbs(), p1.spanning_vertices.cwiseAbs());
}
void Population_density_with_equation::set_ODE(const Advection_diffusion_eqn& adv_diff_eqn) {
    const vector_vector_function advection_velocity_copy = adv_diff_eqn.advection_velocity;
    const auto coupling_velocity_copy = adv_diff_eqn.coupling_velocity;
    const Matrix_type diffusion_coeff = adv_diff_eqn.diffusion_coefficient;
    const int dimension_copy = dimension;
    const value_type& coupling_strength_sum_copy = coupling_strength_sum;
    const double regularizing_factor = lambda;//Tikhonov regularizing factor. 
    const double alpha_copy = alpha;
    adv_diff_eqn_odeint_reformulation adv_diff_eqn_reformulation = [alpha_copy, diffusion_coeff, advection_velocity_copy, dimension_copy, coupling_velocity_copy, regularizing_factor, &coupling_strength_sum_copy](const Center_level_set& X, Center_level_set& dXdt, const value_type t) {
        //Computing "Diffusion velocity" for offcenter:
        const auto regularized_spanning_vertices_decomposition = (X.spanning_vertices * X.spanning_vertices.transpose()+ regularizing_factor*regularizing_factor * Eigen::MatrixXd::Identity(dimension_copy, dimension_copy)).llt();
        dXdt.spanning_vertices = diffusion_coeff * regularized_spanning_vertices_decomposition.solve(X.spanning_vertices);
        dXdt.spanning_vertices = dXdt.spanning_vertices.array();
#ifdef DEBUG
        const auto spanning_vertices_decomposition = X.spanning_vertices.transpose().colPivHouseholderQr();
        dXdt.spanning_vertices = diffusion_coeff * spanning_vertices_decomposition.solve(Eigen::MatrixXd::Identity(dimension_copy, dimension_copy));
        std::cout << "t: " << t << std::endl;
        std::cout << "Center: " << X.center.transpose() << std::endl;
        std::cout << "Center deriv: " << dXdt.center.transpose() << std::endl;
        std::cout << "Spanning vertices (rows): " << std::endl;
        std::cout << X.spanning_vertices.transpose() << std::endl;
        std::cout << "Sanity check: " << std::endl;
        std::cout << (diffusion_coeff * spanning_vertices_decomposition.inverse()).transpose() << std::endl;
        std::cout << "Off center locations.\n";
#endif // DEBUG
        //Compute "coupling velocity":
#ifdef DEBUG
        std::cout << "Coupling velocity: " << coupling_velocity_copy(X.center, coupling_strength_sum_copy).transpose() << std::endl;
#endif // DEBUG
#ifndef AVG_SPEED_LEVELSET
        //Compute "Advection velocity" for center: 
        dXdt.center = advection_velocity_copy(X.center) + coupling_velocity_copy(X.center, coupling_strength_sum_copy);
        for (int j = 0; j < dimension_copy; j++) {
            State_variable v = X.center + alpha_copy * X.spanning_vertices.col(j);
            const State_variable advection_coupling_velocity = (advection_velocity_copy(v) + coupling_velocity_copy(v, coupling_strength_sum_copy) - dXdt.center)/alpha_copy;
            dXdt.spanning_vertices.col(j) += advection_coupling_velocity;
        }
#else
        Matrix_type adv_vel_mat_p = dXdt.spanning_vertices;
        Matrix_type adv_vel_mat_n = adv_vel_mat_p;
        for (int j = 0; j < dimension_copy; j++) {
            const State_variable vp = X.center + alpha_copy * X.spanning_vertices.col(j);
            const State_variable vn = X.center - alpha_copy * X.spanning_vertices.col(j);
            adv_vel_mat_p.col(j) = advection_velocity_copy(vp) + coupling_velocity_copy(vp, coupling_strength_sum_copy);
            adv_vel_mat_n.col(j) = advection_velocity_copy(vn) + coupling_velocity_copy(vn, coupling_strength_sum_copy);
        }
        //center speed is the average of the 2n points
        dXdt.center = X.center;//initial value is not used.
        for (int i = 0; i < dimension_copy; i++) {
            dXdt.center(i) = 0.5 * (adv_vel_mat_p.row(i).mean() + adv_vel_mat_n.row(i).mean());
        }
        for (int j = 0; j < dimension_copy; j++) {
            dXdt.spanning_vertices.col(j) += (adv_vel_mat_p.col(j) - dXdt.center)/alpha_copy;
        }
#endif // !AVG_SPEED_LEVELSET



    };
    adv_diff_eqn_odeint_reformulation adv_diff_eqn_reformulation_coarse_diff = [diffusion_coeff, advection_velocity_copy, dimension_copy, coupling_velocity_copy, regularizing_factor, &coupling_strength_sum_copy](const Center_level_set& X, Center_level_set& dXdt, const value_type t) {
        //A version with diffusion that is independent per direction, used in the stepsize choosing step.
        //Compute "Advection velocity": 
        dXdt.center = advection_velocity_copy(X.center) + coupling_velocity_copy(X.center, coupling_strength_sum_copy);
        dXdt.spanning_vertices = Eigen::MatrixXd::Zero(dimension_copy, dimension_copy);
#define ALL_PT_COARSE_DIFF
#ifdef ALL_PT_COARSE_DIFF
        for (int j = 0; j < dimension_copy; j++) {
            const State_variable offset = X.spanning_vertices.col(j);
            const double offset_norm = offset.norm();
            const double regularized_norm_inverse = offset_norm / (offset_norm * offset_norm + regularizing_factor * regularizing_factor);
            //1D "diffusion velocity approximation: 
            dXdt.spanning_vertices.col(j) += X.spanning_vertices.col(j) * (diffusion_coeff(j, j) / (regularized_norm_inverse * regularized_norm_inverse));
#ifndef AVG_SPEED_LEVELSET
            const State_variable v = X.center + X.spanning_vertices.col(j);

#ifdef DEBUG
            std::cout << "Location " << j << ": " << v.transpose() << std::endl;
            std::cout << "adv_vel: " << advection_velocity_copy(v).transpose() << std::endl;
            std::cout << "dif_vel: " << dXdt.spanning_vertices.col(j).transpose() << std::endl;
#endif // DEBUG
            //const State_variable advection_coupling_velocity = (advection_velocity_copy(v) - dXdt.center);
            const State_variable advection_coupling_velocity = advection_velocity_copy(v) + coupling_velocity_copy(v, coupling_strength_sum_copy) - dXdt.center;
#else
            const State_variable vp = X.center + X.spanning_vertices.col(j);
            const State_variable vn = X.center + X.spanning_vertices.col(j);
            const State_variable advection_coupling_velocity = 0.5*(advection_velocity_copy(vp) + coupling_velocity_copy(vp, coupling_strength_sum_copy) - 
                (advection_velocity_copy(vn) + coupling_velocity_copy(vn, coupling_strength_sum_copy)));
#endif // !AVG_SPEED_LEVELSET
            dXdt.spanning_vertices.col(j) += advection_coupling_velocity;

        }
#endif // ALL_PT_COARSE_DIFF


        //Compute "coupling velocity":
#ifdef DEBUG
        std::cout << "Coupling velocity: " << coupling_velocity_copy(X.center, coupling_strength_sum_copy).transpose() << std::endl;
#endif // DEBUG
    };
    adv_diff_eqn_on_levelset_coarse_diffusion = adv_diff_eqn_reformulation_coarse_diff;
    adv_diff_eqn_on_levelset = adv_diff_eqn_reformulation;
    advection_dynamics = adv_diff_eqn.advection_velocity;
    //The "New thing"
    /*
    const int dimension_copy = dimension;
    const value_type& coupling_strength_sum_copy = coupling_strength_sum;
    adv_diff_eqn_odeint_reformulation adv_diff_eqn_reformulation = [dimension_copy,&coupling_strength_sum_copy,&adv_diff_eqn](const Center_level_set& X, Center_level_set& dXdt, const value_type t) {
        //By reference passing for coupling_strength_sum, since want to set equation once, but different coupling strength at each run. 
        dXdt.center = adv_diff_eqn.advection_velocity(X.center) + adv_diff_eqn.coupling_velocity(X.center, coupling_strength_sum_copy);
        //Computing "Diffusion velocity":
        const auto spanning_vertices_decomposition = X.spanning_vertices.transpose().colPivHouseholderQr();
        dXdt.spanning_vertices = adv_diff_eqn.diffusion_coefficient * spanning_vertices_decomposition.solve(Eigen::MatrixXd::Identity(dimension_copy, dimension_copy));
        for (int j = 0; j < dimension_copy; j++) {
            State_variable v = X.center + X.spanning_vertices.col(j);
            const State_variable advection_coupling_velocity = (adv_diff_eqn.advection_velocity(v) + adv_diff_eqn.coupling_velocity(v, coupling_strength_sum_copy) - dXdt.center);
            dXdt.spanning_vertices.col(j) = dXdt.spanning_vertices.col(j) + advection_coupling_velocity;
        }
    };
    adv_diff_eqn_on_levelset = adv_diff_eqn_reformulation;
    */
}
void Population_density::sort_by_weight() {
    std::sort(p_vect.begin(), p_vect.end(), [](Particle a, Particle b) {
        return a.weight > b.weight;
    });
}
void restrict_to_domain(const Advection_diffusion_eqn& adv_diff_eqn, Center_level_set& center_level_set) {
    //Check whether center of the particle is inside domain. If not, then move it back inside.
    if (!adv_diff_eqn.state_variable_always_valid && !adv_diff_eqn.state_variable_in_domain(center_level_set.center)) {
        center_level_set.center = adv_diff_eqn.state_variable_restrict_to_domain(center_level_set.center);
    }
    //Check wheter "off center" points are in domain. If not, pick the other direction: 
    if (!adv_diff_eqn.state_variable_always_valid)
    {
        for (int j = 0; j < center_level_set.center.size(); j++) {
            if (!adv_diff_eqn.state_variable_in_domain(center_level_set.center + center_level_set.spanning_vertices.col(j))) {
                center_level_set.spanning_vertices.col(j) = -center_level_set.spanning_vertices.col(j);
            }
        }
    }
}
void Population_density_with_equation::update_ODE_adaptive(const value_type timestep, const index_type stepcount) {
    if (dimension != adv_diff_eqn.dimension) {
        std::cout << "Dimension do not match" << std::endl;
        return;
    }
    //set_ODE(adv_diff_eqn);
    namespace odeint = boost::numeric::odeint;
    typedef odeint::runge_kutta_bogacki_shampine32<Center_level_set, value_type, Center_level_set, value_type, odeint::vector_space_algebra> Center_level_set_stepper;
    tbb::concurrent_vector<Center_level_set> center_level_set_vect;
    center_level_set_vect.resize(p_vect.size());
    //Convert Covariance matrix to "level set" form:
    tbb::parallel_for(tbb::blocked_range<int>(0,p_vect.size()),[&](const tbb::blocked_range<int>& index_range){
        for (int i = index_range.begin(); i < index_range.end(); i++) {
            center_level_set_vect[i] = p_vect[i].to_center_level_set();
            restrict_to_domain(adv_diff_eqn, center_level_set_vect[i]);
        }
    });
    std::vector<value_type> coupling_strength_vect;
    coupling_strength_vect.resize(center_level_set_vect.size(), 0.0);
    index_type inf_index = -1;
    for (index_type stepnumber = 0; stepnumber < stepcount; stepnumber++)
    {
        //For each particle: 
        tbb::parallel_for(tbb::blocked_range<int>(0, center_level_set_vect.size()),[&](const tbb::blocked_range<int>& index_range){
            for (int i = index_range.begin(); i < index_range.end(); i++){
                //center_level_set updates: 
                odeint::integrate_adaptive(odeint::make_controlled<Center_level_set_stepper>(1e-5, 1e-5), adv_diff_eqn_on_levelset, center_level_set_vect[i], 0.0, timestep, timestep);
                //convert back to covariance form: 
                const Particle current_particle = center_level_set_vect[i].to_particle(p_vect[i].weight);
                //Find coupling for next step: (part 1: Parallel Transform)
                coupling_strength_vect[i] = adv_diff_eqn.coupling_strength(current_particle, p_vect[i], timestep);
                //Check result: 
                if (std::isinf(current_particle.covariance_matrix.trace())) {
                    inf_index = i;
                    break;
                }
                else
                    p_vect[i] = current_particle;
                
            }
        });
        if (inf_index > -1) {
            std::cout << "Inf at particle " << inf_index;
            //Output the particle causing NaN to workspace
            std::cout << ", Timestep: " << timestep << std::endl;
#ifdef MATLAB_VISUALIZE
            output_particle(inf_index, "Inf_inducing_particle.mat");
#endif // MATLAB_VISUALIZE    
#ifdef _MSC_VER
            throw _FE_OVERFLOW;
#else
            throw EOVERFLOW;
#endif
        }
        //Find coupling for next step: (part 2: Parallel Reduce)
        coupling_strength_sum = 0.0;
        Reduce_sum reduce_sum;
        tbb::parallel_reduce(tbb::blocked_range<std::vector<value_type>::iterator>(coupling_strength_vect.begin(), coupling_strength_vect.end()), reduce_sum);
        coupling_strength_sum = reduce_sum.value;
        //std::cout << "coupling strength sum: " << coupling_strength_sum << std::endl;
    }
}
void Population_density_with_equation::update_ODE_const(const value_type timestep, const index_type stepcount) {
    if (dimension != adv_diff_eqn.dimension) {
        std::cout << "Dimension do not match" << std::endl;
        return;
    }
    //set_ODE(adv_diff_eqn);
    namespace odeint = boost::numeric::odeint;
    typedef odeint::modified_midpoint<Center_level_set, value_type, Center_level_set, value_type, odeint::vector_space_algebra> Center_level_set_constant_stepper;
    tbb::concurrent_vector<Center_level_set> center_level_set_vect;
    center_level_set_vect.resize(p_vect.size());
    //Convert Covariance matrix to "level set" form:
    tbb::parallel_for(tbb::blocked_range<int>(0, p_vect.size()), [&](const tbb::blocked_range<int>& index_range) {
        for (int i = index_range.begin(); i < index_range.end(); i++) {
            center_level_set_vect[i] = p_vect[i].to_center_level_set();
            restrict_to_domain(adv_diff_eqn, center_level_set_vect[i]);
        }
    });
    std::vector<value_type> coupling_strength_vect;
    coupling_strength_vect.resize(center_level_set_vect.size(), 0.0);
    for (index_type stepnumber = 0; stepnumber < stepcount; stepnumber++)
    {
        //For each particle: 
        tbb::parallel_for(tbb::blocked_range<int>(0, center_level_set_vect.size()), [&](const tbb::blocked_range<int>& index_range) {
            for (int i = index_range.begin(); i < index_range.end(); i++) {
                //center_level_set updates: 
                odeint::integrate_const(Center_level_set_constant_stepper(), adv_diff_eqn_on_levelset, center_level_set_vect[i], 0.0, timestep, timestep);
                //convert back to covariance form: 
                const Particle current_particle = center_level_set_vect[i].to_particle(p_vect[i].weight);
                //Find coupling for next step: (part 1: Parallel Transform)
                coupling_strength_vect[i] = adv_diff_eqn.coupling_strength(current_particle, p_vect[i], timestep);
                //Update value in p_vect
                p_vect[i] = current_particle;
            }
        });
        //Find coupling for next step: (part 2: Parallel Reduce)
        coupling_strength_sum = 0.0;
        Reduce_sum reduce_sum;
        tbb::parallel_reduce(tbb::blocked_range<std::vector<value_type>::iterator>(coupling_strength_vect.begin(), coupling_strength_vect.end()), reduce_sum);
        coupling_strength_sum = reduce_sum.value;
    }
}
#ifdef DEBUG_DEADLOCK
std::vector<double> g_time_vect;
#endif // DEBUG_DEADLOCK
void Population_density_with_equation::update_ODE_adaptive_split(const value_type coupling_timestep, const index_type stepcount, const value_type rel_error_bound) {
    this->rel_error_bound = rel_error_bound;
    for (size_t n = 0; n < stepcount; n++)
    {
        //Step 1: parallelly updating original particles. and find coupling caused by these particles. 
        std::vector<value_type> coupling_strength_vect;
        coupling_strength_vect.resize(p_vect.size(),0.0);\

        std::cout << "~ adaptive_split\n";

#ifndef ADAPTIVE_SPLIT_NO_PARALLEL
        tbb::parallel_for(tbb::blocked_range<size_t>(0, p_vect.size()), [&](const tbb::blocked_range<size_t>& index_range) {
            for (int i = index_range.begin(); i < index_range.end(); i++) {
                std::cout << "~ index " + std::to_string(i) + "\n";
                coupling_strength_vect[i] = update_particle_at_index(adv_diff_eqn, coupling_timestep, coupling_timestep, p_vect.begin() + i);
            }
        });
#else
        const int original_particle_count = p_vect.size();
#ifdef DEBUG_DEADLOCK
        g_time_vect.clear();
        g_time_vect.reserve(p_vect.size());
#endif // DEBUG_DEADLOCK

        for (int i = 0; i < original_particle_count; i++) {
            coupling_strength_vect[i] = update_particle_at_index(adv_diff_eqn, coupling_timestep, coupling_timestep, p_vect.begin() + i);
        }
#ifdef DEBUG_DEADLOCK
        MATFile *pmat;
        pmat = matOpen("update_particle_timing.mat", "w");
        mxArray *timing_matlabarray;
        timing_matlabarray = mxCreateDoubleMatrix(1, original_particle_count, mxREAL);
        memcpy(static_cast<double*>(mxGetPr(timing_matlabarray)), g_time_vect.data(), sizeof(double)*original_particle_count);
        matPutVariable(pmat, "timing_array", timing_matlabarray);
        matClose(pmat);
        mxDestroyArray(timing_matlabarray);
#endif // DEBUG_DEADLOCK

#endif // !ADAPTIVE_SPLIT_NO_PARALLEL

        //Step 2: Find coupling for next step: (part 2: Parallel Reduce)
        coupling_strength_sum = 0.0;
        Reduce_sum reduce_sum;
        tbb::parallel_reduce(tbb::blocked_range<std::vector<value_type>::iterator>(coupling_strength_vect.begin(), coupling_strength_vect.end()), reduce_sum);
        coupling_strength_sum = reduce_sum.value;
    }
}
double linear_approx_rel_error(const Particle& x, const State_variable& offset, const Advection_diffusion_eqn& adv_diff_eqn);
std::pair<double, State_variable> particle_linear_approx_rel_error(const Particle x, const Advection_diffusion_eqn& adv_diff_eqn) {
    //Returns the maximum relative error along the axis directions of the Gaussian particle, and the offset direction corresponding to that error. 
    auto svd = x.covariance_matrix.jacobiSvd(Eigen::ComputeFullU);
    const index_type dim = x.center_location.size();
    Particle child_left, child_center, child_right;
    const Matrix_type U = svd.matrixU();
    const State_variable Sigma = svd.singularValues();
    //Check only largest 2 directions (or 1 if problem is 1D)
    const int check_dim = dim > 2 ? 2 : dim;
    std::pair<double, State_variable> max_rel_error = std::make_pair(0.0, State_variable());
    for (index_type col_idx = 0; col_idx < check_dim; col_idx++) {
        const State_variable U_col_idx = U.col(col_idx);
        const State_variable offset = U_col_idx * sqrt(Sigma(col_idx));
        const auto rel_error_val = linear_approx_rel_error(x, offset, adv_diff_eqn);
        if (rel_error_val> max_rel_error.first) {
            max_rel_error.first = rel_error_val;
            max_rel_error.second = offset;
        }
    }
    if (max_rel_error.second.size() != x.center_location.size()) {
        max_rel_error.second = U.col(0) * sqrt(Sigma(0));
    }
    return max_rel_error;
}
#ifdef DEBUG_DEADLOCK
std::tuple<bool, double, State_variable> Population_density_with_equation::update_particle_at_index_single_step_wrapped(const Advection_diffusion_eqn& adv_diff_eqn, const value_type maximum_timestep, particle_vector::iterator itr) {
    return update_particle_at_index_single_step(adv_diff_eqn, maximum_timestep, itr);
    std::mutex m;
    std::condition_variable cv;
    std::tuple<bool, double, State_variable> rval;
    std::thread t([&, adv_diff_eqn, maximum_timestep, itr]() {
        rval = update_particle_at_index_single_step(adv_diff_eqn, maximum_timestep, itr);
        cv.notify_one();
    });
    t.detach();
    std::unique_lock<std::mutex> ulock(m);
    if (cv.wait_for(ulock, std::chrono::seconds(2)) == std::cv_status::timeout) {
        std::cout << "Deadlock inducing particle, at index: " << itr - p_vect.begin() << std::endl;
        std::cout << "x: " << itr->center_location.transpose() << std::endl;
        std::cout << itr->covariance_matrix << std::endl;
        std::cout << "Coupling strength: " << coupling_strength_sum << std::endl;
        std::cout << "Attempted timestep: " << maximum_timestep << std::endl;
        std::cout << "Try again: " << std::endl;
        rval = update_particle_at_index_single_step(adv_diff_eqn, maximum_timestep, itr);
        output_particle(itr - p_vect.begin(), "deadlock_inducing_particle.mat");
        throw std::runtime_error("Timeout");
    }
    return rval;
}
#endif // DEBUG_DEADLOCK

std::tuple<bool, double,State_variable> Population_density_with_equation::update_particle_at_index_single_step(const Advection_diffusion_eqn& adv_diff_eqn, const value_type maximum_timestep, particle_vector::iterator itr) {
    //Check approx validity at initial time step:
    const auto lin_approx_rel_error_prev_state = particle_linear_approx_rel_error(*itr, adv_diff_eqn);

    std::cout << "~ p_vect size: " + std::to_string(p_vect.size()) + "\n";

    namespace odeint = boost::numeric::odeint;
    typedef odeint::runge_kutta_bogacki_shampine32<Center_level_set, value_type, Center_level_set, value_type, odeint::vector_space_algebra> Center_level_set_stepper;
    typedef odeint::runge_kutta_bogacki_shampine32<Center_level_set, value_type, Center_level_set, value_type, odeint::vector_space_algebra> Center_level_set_low_order_stepper;
    //Convert Covariance matrix to "level set" form:
    Center_level_set center_level_set = itr->to_center_level_set();
    restrict_to_domain(adv_diff_eqn, center_level_set);
    double current_timestep = maximum_timestep;
    double ode_stepper_max_timestep = maximum_timestep;//When attempted timestep is too large, Inf will appear in result, causing ode stepper to fail.
    double time_before_split = 0.0;//Can only update to this time before split is needed.
    const auto initial_state = center_level_set;
    const double min_timestep = 5e-3;
    bool can_update_to_end_of_timestep = false;
    bool step_size_found = false;
    Particle p = *itr;
    do
    {
        const auto prev_state = center_level_set;
        //Use a rough estimation to determine if split is needed up to a time step
        odeint::integrate_adaptive(odeint::make_controlled<Center_level_set_low_order_stepper>(1e-2, 1e-2), adv_diff_eqn_on_levelset_coarse_diffusion, center_level_set, 0.0, current_timestep, current_timestep);
        //Check if any Inf or NaN exists. If exists, reduce timestep by half and try evaluation again. 
        bool contain_inf_NaN = false;
        for (int i = 0; i < center_level_set.spanning_vertices.size(); i++) {
            if (std::isnan(center_level_set.spanning_vertices(i)) || std::isinf(center_level_set.spanning_vertices(i))) {
                contain_inf_NaN = true;
                current_timestep *= 0.5;
                ode_stepper_max_timestep = current_timestep;
                center_level_set = prev_state;
                std::cout << "~ NaN found at col " + std::to_string(i) + "\n";
                std::cout << center_level_set.spanning_vertices;
                break;
            }
        }
        if (contain_inf_NaN) {
            continue;
        }
        //check if need split
        p.center_location = center_level_set.center;
        p.covariance_matrix = center_level_set.spanning_vertices * center_level_set.spanning_vertices.transpose();
        const auto lin_approx_rel_error = particle_linear_approx_rel_error(p, adv_diff_eqn);
        if (lin_approx_rel_error.first < rel_error_bound && current_timestep == maximum_timestep) {//Then advanced to end without need to split
            std::cout << "1\n";
            time_before_split += current_timestep;
            can_update_to_end_of_timestep = true;
            step_size_found = true;
        }
        else if (lin_approx_rel_error.first >rel_error_bound) {//Then current timestep is still too large. 
            std::cout << "2\n";
            current_timestep *= 0.5;
            center_level_set = prev_state;
        }
        else //!need_split && current_timestep < maximum_timestep
        {
            std::cout << "3\n";
            time_before_split += current_timestep;
            //Check with less error bound: 
            if (lin_approx_rel_error.first > 0.8 * rel_error_bound) {//Split if error is close to its bound
                step_size_found = true;
            }
            else {
                current_timestep *= 0.5;//try advance further
            }
        }
    } while (!step_size_found && min_timestep < current_timestep);
    if (time_before_split == 0.0) {
        time_before_split += current_timestep;
    }

    Center_level_set center_level_set_precise = initial_state;
#ifdef DEBUG_DEADLOCK
    //printf("D\n");
#endif // DEBUG_DEADLOCK
    odeint::integrate_adaptive(odeint::make_controlled<Center_level_set_stepper>(1e-2, 1e-2), adv_diff_eqn_on_levelset, center_level_set_precise, 0.0, time_before_split, ode_stepper_max_timestep);
    itr->center_location = center_level_set_precise.center;
    itr->covariance_matrix = center_level_set_precise.spanning_vertices * center_level_set_precise.spanning_vertices.transpose();
    if (can_update_to_end_of_timestep)
    {
#ifdef DEBUG_DEADLOCK
        //printf("E1\n");
#endif // DEBUG_DEADLOCK
        return std::make_tuple(true, 0.0, State_variable());
    }
    else
    {
#ifdef DEBUG_DEADLOCK
        //printf("E2\n");
#endif // DEBUG_DEADLOCK
        const auto lin_approx_rel_error = particle_linear_approx_rel_error(*itr, adv_diff_eqn);
#ifdef DEBUG_DEADLOCK
        if (lin_approx_rel_error.second.size() != dimension) {
            std::cout << "Size mismatch\n";
        }
        //printf("E2_2\n");
#endif // DEBUG_DEADLOCK
        return std::make_tuple(false, maximum_timestep - time_before_split, lin_approx_rel_error.second);
    }
}
std::array<Particle, 3> split_particle_in_direction(const Particle x, const State_variable offset);
value_type Population_density_with_equation::update_particle_at_index(const Advection_diffusion_eqn& adv_diff_eqn, const value_type timestep, const value_type coupling_timestep, particle_vector::iterator itr) {
    const Particle prev_state = *itr;
#ifdef DEBUG_DEADLOCK
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    const auto update_result = update_particle_at_index_single_step_wrapped(adv_diff_eqn, timestep, itr);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    typedef std::chrono::duration<double, std::ratio<1, 1>> seconds;
    g_time_vect.push_back(seconds(t1 - t0).count());
#else
    const auto update_result = update_particle_at_index_single_step(adv_diff_eqn, timestep, itr);
#endif

    std::cout << "~ results: " + std::to_string(std::get<0>(update_result)) + " " + std::to_string(std::get<1>(update_result)) + "\n";
    std::cout << "~ particle weight: " + std::to_string(itr->weight) + "\n";

    //(updated to end, time remaining, split offset)
    const auto remaining_time = std::get<1>(update_result);
    if (std::get<0>(update_result)) {
        //updated to end of timestep. Need to find coupling. 
        const auto coupling_strength = adv_diff_eqn.coupling_strength(*itr, prev_state, coupling_timestep);
        return adv_diff_eqn.coupling_strength(*itr, prev_state, coupling_timestep);
    }
    else {
        //updated to time to split. Need to 
        //1. find coupling up to now:
        value_type coupling_before_split = 0.0;
        if (timestep - std::get<1>(update_result) > 0.0)
        {
            coupling_before_split = adv_diff_eqn.coupling_strength(*itr, prev_state, coupling_timestep);
        }
        //Low weight relaxed_treatment:
        if (itr->weight < split_relax_weight) {
            itr->covariance_matrix /= 2;
            std::cout << "~ low weight\n";
            return coupling_before_split + 0;//update_particle_at_index(adv_diff_eqn, remaining_time, coupling_timestep, itr);
        }
        //2. Split particles. 
        const auto particle_children = split_particle_in_direction(*itr, std::get<2>(update_result));
        //3. Store children:
        *itr = particle_children[0];
        auto left_child_itr = p_vect.push_back(particle_children[1]);
        auto right_child_itr = p_vect.push_back(particle_children[2]);
        //4. Find rval iteratively

        std::cout << "~ remaining time: " + std::to_string(remaining_time) + "\n";

        return coupling_before_split +
            update_particle_at_index(adv_diff_eqn, remaining_time, coupling_timestep, itr) +
            update_particle_at_index(adv_diff_eqn, remaining_time, coupling_timestep, left_child_itr) +
            update_particle_at_index(adv_diff_eqn, remaining_time, coupling_timestep, right_child_itr);
    }
}
void approximate_jacobian(Matrix_type& jacobian, const vector_vector_function& full_derivative, const State_variable& x) {
    //dF/dxi found by (F(X+eps*e_i) - F(X-eps*e_i))/(2*eps)
    const value_type eps = 1e-4;
    for (int col_index = 0; col_index < x.size(); col_index++) {
        auto xpei = x;
        xpei(col_index) += eps;
        auto xmei = x;
        xmei(col_index) -= eps;
        jacobian.col(col_index) = (full_derivative(xpei) - full_derivative(xmei)) / (2.0*eps);
    }
}
Matrix_type approximate_jacobian(const vector_vector_function& full_derivative, const State_variable& x) {
    //dF/dxi found by (F(X+eps*e_i) - F(X-eps*e_i))/(2*eps)
    Matrix_type jacobian = Eigen::MatrixXd::Zero(x.size(), x.size());
    approximate_jacobian(jacobian, full_derivative, x);
    return jacobian;
}

std::array<Particle, 3> split_particle_in_direction(const Particle x, const State_variable offset) {
    std::array<Particle, 3> rval;
    rval.fill(x);
#ifdef DEBUG
    std::cout << "singular value before: "<< x.covariance_matrix.jacobiSvd().singularValues().transpose() << std::endl;
#endif // DEBUG

    const value_type weight_noncenter_child_ratio = 0.21921;//WAS 0.213109;
    const value_type weight_center_child_ratio = 1.0 - 2 * weight_noncenter_child_ratio;
    const value_type distance_ratio = 1.03332;//a = UPDATE: 1.46133, w=0.21921
    //const value_type distance_ratio = 1.040621;//WAS 1.47166, changed to 1.47166/sqrt(2) due to a change in form of kernel function (2pi vs 4pi);
    const Matrix_type direction_outer_product = offset * offset.transpose();
    //left child: 
    rval[0].weight *= weight_noncenter_child_ratio;
    rval[0].center_location -= distance_ratio * offset;
    rval[0].covariance_matrix -= 0.5*direction_outer_product;
    //center child: 
    rval[1].weight *= weight_center_child_ratio;
    rval[1].covariance_matrix -= 0.5*direction_outer_product;
    //right child: 
    rval[2].weight *= weight_noncenter_child_ratio;
    rval[2].center_location += distance_ratio * offset;
    rval[2].covariance_matrix -= 0.5*direction_outer_product;
#ifdef DEBUG
    //Check values: 
    const value_type lval = rval[0].density_at(rval[1].center_location);
    const value_type cval = rval[1].density_at(rval[1].center_location);
    const value_type c_origin = x.density_at(rval[1].center_location);
    std::cout << "Split offset: " << offset.transpose() << std::endl;
    std::cout << "Norm of left child: " << rval[0].center_location.norm() << std::endl;
    std::cout << "Contribution at center: " << lval << std::endl;
    std::cout << "Contribution of center child at center: " << cval << std::endl;
    std::cout << "Original density at center: " << c_origin << std::endl;
    std::cout << "Relative error introduced: " << (cval + 2 * lval) / c_origin - 1 << std::endl;
    std::cout << "singular value after: " << rval[0].covariance_matrix.jacobiSvd().singularValues().transpose() << std::endl;
#endif // DEBUG
    return rval;
}
std::array<Particle, 3> split_particle_in_direction(const Particle x, const State_variable direction, const value_type singular_value) {
    //Direction is a column unit vector.
    return split_particle_in_direction(x, direction * sqrt(singular_value));
}
double linear_approx_rel_error(const Particle& x, const State_variable& offset, const Advection_diffusion_eqn& adv_diff_eqn) {
    const State_variable center_derivative = adv_diff_eqn.advection_velocity(x.center_location);
    if (center_derivative.norm() < 1e-9) {//Avoid checking if derivative at center is 0.
        return 0.0;
    }
    const State_variable near_center = x.center_location + offset;
    const State_variable far_center = x.center_location + 2 * offset;
    const State_variable far_center_deriv = (adv_diff_eqn.advection_velocity(far_center) - center_derivative)*0.5;
    const State_variable near_center_deriv = adv_diff_eqn.advection_velocity(near_center) - center_derivative;
    const double rel_error = (near_center_deriv - far_center_deriv).norm() / center_derivative.norm();
    return rel_error;
}
void Population_density_with_equation::check_linear_approx(const int particle_index) const {
    const Particle& target_particle = p_vect[particle_index];
    auto svd_structure = target_particle.covariance_matrix.jacobiSvd(Eigen::ComputeFullU);
    const Matrix_type covariance_sqrt = svd_structure.matrixU();
    const State_variable eigenvalues = svd_structure.singularValues().cwiseSqrt();

    //Find derivatives: 
    const State_variable center_derivative = adv_diff_eqn.advection_velocity(target_particle.center_location);
    for (int i = 0; i < dimension; i++) {
        const State_variable offset = covariance_sqrt.col(i).array()*sqrt(1)*eigenvalues(i);
        const State_variable near_center = target_particle.center_location + offset;
        const State_variable far_center = target_particle.center_location + 2 * offset;
        const State_variable far_center_deriv = (adv_diff_eqn.advection_velocity(far_center) - center_derivative)*0.5;
        const State_variable near_center_deriv = adv_diff_eqn.advection_velocity(near_center) - center_derivative;
        std::cout << "offset: " << offset.transpose() << std::endl;
        std::cout << "Eigenvalue: " << eigenvalues(i) << std::endl;
        std::cout << "Near center: ";
        std::cout << near_center_deriv.transpose() << std::endl;
        std::cout << "Far from center: ";
        std::cout << far_center_deriv.transpose() << std::endl;
        std::cout << "In " << i + 1 << "th eigen direction, derivative is off linear by " << (near_center_deriv - far_center_deriv).norm() / near_center_deriv.norm() << std::endl;
        std::cout << "relative error versus derivative is " << (near_center_deriv - far_center_deriv).norm() / adv_diff_eqn.advection_velocity(near_center).norm() << std::endl;
    }
}
bool need_split_in_direction(const Particle& x, const State_variable& offset, const Advection_diffusion_eqn& adv_diff_eqn, const value_type rel_error_bound) {
    const double rel_error = linear_approx_rel_error(x,offset,adv_diff_eqn);
#ifdef DEBUG
    std::cout << "offset: " << offset.transpose() << std::endl;
    std::cout << "rel_error: " << rel_error << std::endl;
#endif // DEBUG

    if (rel_error > rel_error_bound)
        return true;
    return false;
}
std::vector<Particle> split_particle(const Particle x, const Advection_diffusion_eqn& adv_diff_eqn, const value_type rel_error_bound, const value_type split_relax_weight) {
    std::vector<Particle> rval;
    //Small weight: shrink instead of split:
    if (x.weight < split_relax_weight) {
        rval.push_back(x);
        rval[0].covariance_matrix = rval[0].covariance_matrix / 2;
        return rval;
    }
    //Checked output okay.
    auto svd_structure = x.covariance_matrix.jacobiSvd(Eigen::ComputeFullU);
    const index_type dim = x.center_location.size();
    Particle child_left, child_center, child_right;
    const Matrix_type U = svd_structure.matrixU();
    const State_variable Sigma = svd_structure.singularValues();
    rval.reserve(3);
    rval.insert(rval.end(), x);
    for (index_type col_idx = 0; col_idx < dim; col_idx++) {
        const State_variable U_col_idx = U.col(col_idx);
        const State_variable offset = U_col_idx * sqrt(Sigma(col_idx));
        if (!need_split_in_direction(x,offset,adv_diff_eqn, rel_error_bound))
            break;
        const index_type current_size = rval.size();//Should be pow(3,i)
        rval.resize(3 * current_size);
        for (index_type i = 0; i < current_size; i++) {
            const auto particle_tuple = split_particle_in_direction(rval[i],U_col_idx,Sigma(col_idx));
            rval[i] = particle_tuple[1];
            rval[current_size + i] = particle_tuple[0];
            rval[2 * current_size + i] = particle_tuple[2];
        }
    }
    return rval;
}
std::vector<Particle> split_particle_old(const Particle x, const value_type tau) {
    auto svd_structure = x.covariance_matrix.jacobiSvd(Eigen::ComputeFullU);
    std::vector<Particle> rval;
    const index_type dim = x.center_location.size();
    Particle child_left, child_center, child_right;
    if (svd_structure.singularValues().operator()(0) < 2.0 * sqrt(tau)) {//Then split is not necessary.
        return rval;
    }
    const Matrix_type U = svd_structure.matrixU();
    State_variable Sigma = svd_structure.singularValues();
    rval.reserve(3);
    rval.insert(rval.end(), x);
    for (index_type col_idx = 0; col_idx < dim; col_idx++) {
        if (Sigma(col_idx) < 2.0)
            break;
        const index_type current_size = rval.size();//Should be pow(3,i)
        rval.resize(3 * current_size);
        const State_variable U_col_idx = U.col(col_idx);
        for (index_type i = 0; i < current_size; i++) {
            const auto particle_tuple = split_particle_in_direction(rval[i], U_col_idx, Sigma(col_idx));
            rval[i] = particle_tuple[1];
            rval[current_size + i] = particle_tuple[0];
            rval[2 * current_size + i] = particle_tuple[2];
        }
    }
    return rval;
}
struct Indexed_value {
    value_type value;
    index_type index;
    bool operator<(const Indexed_value& rhs) const{
        return value < rhs.value;
    }
};
void Population_density::sort_in_nth_coordinate(const index_type coordinate) {
    std::sort(p_vect.begin(), p_vect.end(), [coordinate](const Particle &lhs, const Particle &rhs) {
        return lhs.center_location[coordinate] < rhs.center_location[coordinate];
    });
}
std::vector<index_type> Population_density::sort_index_in_nth_coordinate(const index_type coordinate) const{
    auto nth_coordinate = get_location_in_dimension_n(coordinate);
    std::vector<Indexed_value> indexed_value;
    indexed_value.resize(nth_coordinate.size());
    for (int i = 0; i < nth_coordinate.size(); i++) {
        indexed_value[i].index = i;
        indexed_value[i].value = nth_coordinate[i];
    }
    std::sort(indexed_value.begin(), indexed_value.end());
    std::vector<index_type> rval;
    rval.resize(nth_coordinate.size());
    for (int i = 0; i < nth_coordinate.size(); i++) {
        rval[i] = indexed_value[i].index;
    }
    return rval;
}
void Population_density_with_equation::combine_particles() {
#ifdef DISPLAY_COMBINE
    std::cout << "Size before: " << p_vect.size() << std::endl;  
#endif // DISPLAY_COMBINE
    value_type weight_sum = 0.0;
    value_type max_weight = 0.0;
    for (const Particle &x : p_vect) {
        weight_sum += x.weight;
        if (x.weight > max_weight)
            max_weight = x.weight;
    }
    const value_type combine_weight_lowerbound = 1e-3*max_weight < 1e-5*weight_sum ? 1e-3*max_weight : 1e-5*weight_sum;
    //PART A: COMBINE PARTICLES THAT ARE CLOSE. 
    const value_type distance_factor = sqrt(tau);
    const value_type radius_threhold = 1.0 * distance_factor;//In split_particles, closest distance is 1.47 * sqrt(singular value">2" *tau)
    //NOTE: What folllows below is a not very optimized version for proof of concept. 
    const value_type grid_block_size = 4.0 * radius_threhold;
    std::unordered_map<long long int, std::vector<int>> grid_id_to_index_map;
    //Find min and max in each dimension: 
    auto min_center_location = p_vect[0].center_location;
    auto max_center_location = p_vect[0].center_location;
    for (Particle &x : p_vect) {
        for (int i = 0; i < dimension; i++) {
            if (x.center_location[i] < min_center_location[i])
                min_center_location[i] = x.center_location[i];
            if (x.center_location[i] > max_center_location[i])
                max_center_location[i] = x.center_location[i];
        }
    }
    //Decide number of grid blocks in each dimension: 
    std::vector<int> block_count;
    for (int i = 0; i < dimension; i++) {
        block_count.push_back((max_center_location[i] - min_center_location[i]) / grid_block_size + 2);
    }
    std::function<long long int(const State_variable&)> State_variable_to_linear_grid_index = [block_count, grid_block_size, min_center_location](const State_variable& center_location) {
        const int dimension = center_location.size();
        long long int rval = 0LL;
        const auto difference = center_location - min_center_location;
        for (int i = 0; i < dimension; i++) {
            if (i >= 1)
                rval *= (long long int)block_count[i - 1];
            rval += (long long int)(difference[i] / grid_block_size);
        }
        return rval;
    };
    std::vector<std::vector<int>*> grids_with_particles;
    for (int i = 0; i < p_vect.size(); i++) {
        long long int linear_grid_index = State_variable_to_linear_grid_index(p_vect[i].center_location);
        const auto matching_grid = grid_id_to_index_map.find(linear_grid_index);
        if (matching_grid != grid_id_to_index_map.end()) {
            //An entry already exists in that grid.
            //Then, add particle to the corresponding term: 
            matching_grid->second.push_back(i);
        }
        else {
            //This grid has a particle inside for the first time
            grid_id_to_index_map.insert(std::make_pair(linear_grid_index, std::vector<int> {i}));
            grids_with_particles.push_back(&(grid_id_to_index_map.find(linear_grid_index)->second));
        }
    }
    //Parallel in each grid block: 
    particle_vector p_vect_next;
    tbb::parallel_for(tbb::blocked_range<int>(0, grids_with_particles.size()), [&](const tbb::blocked_range<int>& block_index_bound) {
        for (int block_index = block_index_bound.begin(); block_index < block_index_bound.end(); block_index++) {
            const int local_size = grids_with_particles[block_index]->size();
            std::vector<bool> combined(local_size, false);
            for (int local_index = 0; local_index < local_size; local_index++) {
                //iterate through all particles in block: 
                if (combined[local_index]) {
                    //Combined once already
                    continue;
                }
                combined[local_index] = true;
                std::vector<int> combine_particle_indices;
                combine_particle_indices.push_back(grids_with_particles[block_index]->at(local_index));
                for (int search_index = local_index + 1; search_index < local_size; search_index++) {
                    if (combined[search_index])
                        continue;
                    const State_variable center_difference = p_vect[grids_with_particles[block_index]->at(search_index)].center_location - p_vect[grids_with_particles[block_index]->at(local_index)].center_location;
                    if (center_difference.norm() < radius_threhold || p_vect[grids_with_particles[block_index]->at(search_index)].weight<combine_weight_lowerbound && center_difference.norm() < 4.0*radius_threhold) {
                        combined[search_index] = true;
                        combine_particle_indices.push_back(grids_with_particles[block_index]->at(search_index));
                    }
                }
                Particle combined_particle;
                combined_particle = combine_particle_at_indices(combine_particle_indices);
                p_vect_next.push_back(combined_particle);
            }
        }
    });
    p_vect = p_vect_next;
#ifdef DISPLAY_COMBINE
    std::cout << "Size after: " << p_vect.size() << std::endl;
#endif // DISPLAY_COMBINE

    return;
}
Particle Population_density_with_equation::combine_particle_at_indices(const std::vector<index_type> &indices) const{//Returns the index of remaining 
    if (indices.size() == 1) {
        return p_vect[indices[0]];
    }
    Particle combined_particle(dimension);
    combined_particle.covariance_matrix *= 0;
    value_type weight_total = 0.0;
    State_variable center_sum = Eigen::VectorXd::Zero(dimension);
    for (index_type i = 0; i < indices.size(); i++) {
        const auto &x = p_vect[indices[i]];
        weight_total += x.weight;
        center_sum += x.weight * x.center_location;
    }
    combined_particle.weight = weight_total;
    combined_particle.center_location = center_sum / weight_total;
    for (index_type i = 0; i < indices.size(); i++) {
        const auto &x = p_vect[indices[i]];
        const State_variable center_difference = (x.center_location - combined_particle.center_location);
        combined_particle.covariance_matrix += x.weight / weight_total * (center_difference * center_difference.transpose() + x.covariance_matrix);
    }
    return combined_particle;
}
void Population_density_with_equation::split_particles(const value_type rel_error_bound) {
    const int prev_size = size();
    tbb::parallel_for(tbb::blocked_range<int>(0,p_vect.size()),[&](tbb::blocked_range<int>& index_range)
    {
        for (int i = index_range.begin(); i < index_range.end(); i++) {
            if (std::isnan(this->operator[](i).covariance_matrix.trace())) {
                std::cout << "NaN at particle " << i << " center: " << this->operator[](i).center_location.transpose();
                std::cout << "\nCovariance: \n" << this->operator[](i).covariance_matrix << std::endl;
            }
            const auto splited_particle_vector = split_particle(this->operator[](i), adv_diff_eqn, rel_error_bound,split_relax_weight);
            p_vect[i] = splited_particle_vector[0];
            if (splited_particle_vector.size() >= 2) {
                p_vect.grow_by(splited_particle_vector.begin() + 1, splited_particle_vector.end());
            }
        }
    });
}
void Population_density_with_equation::uniform_shift(const State_variable& pertubation) {
    tbb::parallel_for(tbb::blocked_range<int>(0, p_vect.size()), [&](tbb::blocked_range<int>& index_range) 
    {
            for (int i = index_range.begin(); i < index_range.end(); i++) {
                p_vect[i].center_location = p_vect[i].center_location + pertubation;
        }
    });
}
std::vector<double> Population_density::get_location_in_dimension_n(const index_type n) const{
    std::vector<double> rval;
    rval.reserve(p_vect.size());
    for (auto &x : p_vect) {
        rval.push_back(x.center_location[n]);
    }
    return rval;
}
std::vector<double> Population_density::get_weight() const {
    std::vector<double> rval;
    rval.reserve(p_vect.size());
    for (auto &x : p_vect) {
        rval.push_back(x.weight);
    }
    return rval;
}
#ifdef MATLAB_VISUALIZE
Engine* global_matlab_engine = NULL;
#endif
index_type closest_grid_index(const value_type x, const value_type y, const value_type x_lb, const value_type x_ub, const value_type y_lb, const value_type y_ub, const index_type spatial_resolution) {
    index_type i = (x - x_lb) / (x_ub - x_lb) * spatial_resolution + 0.5;
    index_type j = (y - y_lb) / (y_ub - y_lb) * spatial_resolution + 0.5;
    i = i < 0 ? 0 : i;
    i = i > spatial_resolution - 1 ? spatial_resolution - 1 : i;
    j = j < 0 ? 0 : j;
    j = j > spatial_resolution - 1 ? spatial_resolution - 1 : j;
    return i * spatial_resolution + j;
}
#ifdef MATLAB_VISUALIZE
void passplotcommand(Plot_handle plot_handle, const char* argument) {
    engEvalString(plot_handle, argument);
}
#endif // MATLAB_VISUALIZE

std::pair<value_type, value_type>plot_index_range(const value_type x, const value_type lb, const value_type ub, const index_type grid_count, const value_type tau) {
    //Seems to be a function determining particles that could be within this region. tau is still preserved
    const value_type sigma = sqrt(2.0 * 2.0 * tau);//"Actual t" <= 2*tau.
    index_type i_lb = (x - lb - 2.0*sigma) / (ub - lb) * grid_count;
    i_lb = i_lb < 0 ? 0 : i_lb;
    i_lb = i_lb > grid_count ? grid_count : i_lb;
    index_type i_ub = (x - lb + 2.0*sigma) / (ub - lb) * grid_count + 2.0;
    i_ub = i_ub < 0 ? 0 : i_ub;
    i_ub = i_ub > grid_count ? grid_count : i_ub;
    return std::pair<value_type, value_type>(i_lb, i_ub);
}
int find_index_less_than_x(const value_type x, const value_type x_lb, const int total_point_count, const double stepsize) {
    if (x < x_lb) {
        return 0;
    }
    if (x > x_lb + (total_point_count - 1)*stepsize) {
        return total_point_count - 1;
    }
    return int((x - x_lb) / stepsize);
}
int find_index_greater_than_x(const value_type x, const value_type x_lb, const int total_point_count, const double stepsize) {
    if (x < x_lb) {
        return 0;
    }
    if (x > x_lb + (total_point_count - 1)*stepsize) {
        return total_point_count - 1;
    }
    return int((x - x_lb) / stepsize) + 1;
}
double density_projection_at(const Particle& p, const int coord_idx, const double x, const double smoothing_factor) {
    const double mu = p.center_location(coord_idx);
    const double sigma = sqrt(p.covariance_matrix(coord_idx, coord_idx)) + smoothing_factor;//To make the density non-vanishing, 
    const double normalized_x = (x - mu) / sigma;
    return p.weight / sqrt(2 * pi) * exp(-normalized_x * normalized_x / 2);
}
double Population_density::average_in_index(const int coord_idx) const {
    double total_weight = 0.0;
    double weighted_sum = 0.0;
    for (const Particle& p : p_vect) {
        total_weight += p.weight;
        weighted_sum += p.weight * p.center_location(coord_idx);
    }
    return weighted_sum / total_weight;
}
#ifdef MATLAB_VISUALIZE
Plot_handle Population_density::plot_density(std::vector<bool> projection_dimensions, const value_type x_lb, const value_type x_ub, const value_type y_lb, const value_type y_ub, const char *imagesc_options) const {
    //Parse 1 dimensional plot as well: 
    if (projection_dimensions.size() == dimension && std::count(projection_dimensions.begin(), projection_dimensions.end(), true) == 1) {
        return plot_density(std::distance(std::find(projection_dimensions.begin(), projection_dimensions.end(), true), projection_dimensions.begin()), x_lb, x_ub, imagesc_options);
    }
    if (projection_dimensions.size() != dimension || std::count(projection_dimensions.begin(), projection_dimensions.end(), true) != 2) {
        fprintf(stderr, "\nNot a valid projection onto a 2-dimension subspace.\n");
        return global_matlab_engine;
    }
    if (global_matlab_engine == NULL) {
        global_matlab_engine = engOpen("");
        if (!global_matlab_engine) {
            fprintf(stderr, "\nCan't start MATLAB engine\n");
            return global_matlab_engine;
        }
    }
    engEvalString(global_matlab_engine, "cla");
    index_type first_coor, second_coor;
    bool first_find = false;
    for (index_type i = 0; i < dimension; i++) {
        if (projection_dimensions[i])
            if (first_find)
                second_coor = i;
            else {
                first_coor = i;
                first_find = true;
            }
    }
    const int spatial_resolution = 256;
    std::vector<State_variable> plot_location(spatial_resolution*spatial_resolution, p_vect[0].center_location);
    const value_type x_step = (x_ub - x_lb) / spatial_resolution;
    const value_type y_step = (y_ub - y_lb) / spatial_resolution;
    //Setup grid
    tbb::parallel_for(tbb::blocked_range<int>(0,spatial_resolution),[&](tbb::blocked_range<int>& range_bound)
    {
        for (int j = range_bound.begin(); j < range_bound.end(); j++) {
            for (int i = 0; i < spatial_resolution; i++) {
                plot_location[i*spatial_resolution + j][first_coor] = x_lb + x_step * i;
                plot_location[i*spatial_resolution + j][second_coor] = y_lb + y_step * j;
            }
        }
    });
    tbb::concurrent_vector<value_type> density_vec(spatial_resolution*spatial_resolution,0.0);
    tbb::parallel_for(tbb::blocked_range<int>(0, p_vect.size()), [&](const tbb::blocked_range<int>& index_range)
    {
        for (int particle_index = index_range.begin(); particle_index < index_range.end(); particle_index++) {
            const Particle& p = p_vect[particle_index];
            auto svd_structure = p.covariance_matrix.jacobiSvd();
            const double largest_stdv = sqrt(svd_structure.singularValues().operator()(0));
            const double p_x = p.center_location[first_coor];
            const double p_y = p.center_location[second_coor];
            const int x_index_lb = find_index_less_than_x(p_x - 3.5 * largest_stdv, x_lb, spatial_resolution, x_step);
            const int x_index_ub = find_index_greater_than_x(p_x + 3.5 * largest_stdv, x_lb, spatial_resolution, x_step);
            const int y_index_lb = find_index_less_than_x(p_y - 3.5 * largest_stdv, y_lb, spatial_resolution, y_step);
            const int y_index_ub = find_index_greater_than_x(p_y + 3.5 * largest_stdv, y_lb, spatial_resolution, y_step);
            for (int j = y_index_lb; j <= y_index_ub; j++)
                for (int i = x_index_lb; i <= x_index_ub; i++)
                    density_vec[i*spatial_resolution + j] += p.density_projection_at_coordinate(plot_location[i*spatial_resolution + j], projection_dimensions);
        }
    });
    std::vector<value_type> x_vec(spatial_resolution), y_vec(spatial_resolution);
    for (int i = 0; i < spatial_resolution; i++) {
        x_vec[i] = i * x_step + x_lb;
        y_vec[i] = i * y_step + y_lb;
    }
    mxArray *density_matlabarray;
    mxArray *x_matlabarray, *y_matlabarray;
    density_matlabarray = mxCreateDoubleMatrix(spatial_resolution, spatial_resolution, mxREAL);
    x_matlabarray = mxCreateDoubleMatrix(1, spatial_resolution, mxREAL);
    y_matlabarray = mxCreateDoubleMatrix(1, spatial_resolution, mxREAL);
    std::copy(density_vec.begin(), density_vec.end(), mxGetPr(density_matlabarray));
    std::copy(x_vec.begin(), x_vec.end(), mxGetPr(x_matlabarray));
    std::copy(y_vec.begin(), y_vec.end(), mxGetPr(y_matlabarray));
    engPutVariable(global_matlab_engine, "x", x_matlabarray);
    engPutVariable(global_matlab_engine, "y", y_matlabarray);
    engPutVariable(global_matlab_engine, "w", density_matlabarray);
    if (p_vect.size() >= FAST_PLOT_PARTICLE_COUNT) {
        engEvalString(global_matlab_engine, "h = [0.25 0.5 0.25];H=h'*h;w=imfilter(w,H);");
    }
    char imagesc_arg[256];
    sprintf(imagesc_arg, "pcolor(x,y,w);caxis(%s);shading interp;colorbar;colormap(jet(256))", imagesc_options);
    engEvalString(global_matlab_engine, imagesc_arg);
    engEvalString(global_matlab_engine, "set(gca,'YDir','normal')");
    engEvalString(global_matlab_engine, "drawnow");
    mxDestroyArray(density_matlabarray);
    mxDestroyArray(x_matlabarray);
    mxDestroyArray(y_matlabarray);
    return global_matlab_engine;
}
Plot_handle Population_density::plot_density(const int projection_dimension, const value_type x_lb, const value_type x_ub, const char *imagesc_options) const{
    //project into 1 dimension
    const int spatial_resolution = 100;
    const double spatial_stepsize = (x_ub - x_lb) / (spatial_resolution - 1);
    std::vector<double> x_vec;
    x_vec.resize(spatial_resolution);
    std::vector<double> rho_vec;
    rho_vec.resize(spatial_resolution);
    tbb::parallel_for(tbb::blocked_range<int>(0, spatial_resolution), [&](tbb::blocked_range<int>&) {
        for (int i = 0; i < spatial_resolution; i++) {
            x_vec[i] = x_lb + i * spatial_stepsize;
            rho_vec[i] = 0;
            for (const Particle& x : p_vect) {
                rho_vec[i] += density_projection_at(x, projection_dimension, x_vec[i], 2*spatial_stepsize);
            }
        }
    });
    mxArray *x_matlabarray;
    x_matlabarray = mxCreateDoubleMatrix(1, spatial_resolution, mxREAL);
    std::copy(x_vec.begin(), x_vec.end(), mxGetPr(x_matlabarray));
    engPutVariable(global_matlab_engine, "x", x_matlabarray);
    mxArray *rho_matlabarray;
    rho_matlabarray = mxCreateDoubleMatrix(1, spatial_resolution, mxREAL);
    std::copy(rho_vec.begin(), rho_vec.end(), mxGetPr(rho_matlabarray));
    engPutVariable(global_matlab_engine, "y", rho_matlabarray);
    engEvalString(global_matlab_engine, "plot(x,y);");
    std::string matlabarg;
    matlabarg = "xlim([";
    matlabarg.append(std::to_string(x_lb));
    matlabarg.append(", ");
    matlabarg.append(std::to_string(x_ub));
    matlabarg.append("]);");
    engEvalString(global_matlab_engine, matlabarg.c_str());
    return global_matlab_engine;
}
Plot_handle Population_density::plot_density(const value_type x_lb, const value_type x_ub, const value_type y_lb, const value_type y_ub, const char *imagesc_options) const{
    std::vector<bool> range_dimensions(dimension, false);
    range_dimensions[0] = range_dimensions[1] = true;
    return plot_density(range_dimensions, x_lb, x_ub, y_lb, y_ub, imagesc_options);
}
Plot_handle Population_density::plot(std::vector<bool> projection_dimensions, const char* plot_options, const int plot_flags){
    const int dimension_count = std::count(projection_dimensions.begin(), projection_dimensions.end(), true);
    int matlab_array_size = g_plot_max_size_passed > size() ? size() : g_plot_max_size_passed;
    bool delete_remaining = g_plot_max_size_passed < p_vect.size();
    sort_by_weight();
    if (global_matlab_engine == NULL) {
        global_matlab_engine = engOpen("");
        if (!global_matlab_engine) {
            fprintf(stderr, "\nCan't start MATLAB engine\n");
            return global_matlab_engine;
        }
    }
    //Parse plot_flag
    bool skip_first = bool(plot_flags / 4);
    bool skip_second = bool(plot_flags % 4 / 2);
    bool overwrite_previous = bool(plot_flags % 2);
    if (dimension_count == 2)
    {
        index_type first_coor, second_coor;
        bool first_find = false;
        for (index_type i = 0; i < dimension; i++) {
            if (projection_dimensions[i])
                if (first_find)
                    second_coor = i;
                else {
                    first_coor = i;
                    first_find = true;
                }
        }
        //Plot position: 
        auto x_vec = get_location_in_dimension_n(first_coor);
        auto y_vec = get_location_in_dimension_n(second_coor);
        auto weight_vec = get_weight();
        if (delete_remaining) {
            x_vec.resize(g_plot_max_size_passed);
            y_vec.resize(g_plot_max_size_passed);
            weight_vec.resize(g_plot_max_size_passed);
        }
        mxArray *x_matlabarray, *y_matlabarray, *weight_matlabarray;
        x_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        std::copy(x_vec.begin(), x_vec.end(), mxGetPr(x_matlabarray));
        y_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        std::copy(y_vec.begin(), y_vec.end(), mxGetPr(y_matlabarray));
        weight_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        std::copy(weight_vec.begin(), weight_vec.end(), mxGetPr(weight_matlabarray));
        engPutVariable(global_matlab_engine, "x", x_matlabarray);
        engPutVariable(global_matlab_engine, "y", y_matlabarray);
        engPutVariable(global_matlab_engine, "w", weight_matlabarray);
        if (overwrite_previous) {
            engEvalString(global_matlab_engine, "cla");
        }
        engEvalString(global_matlab_engine, "scatter(x,y,10000*w,'MarkerEdgeColor',[1 0 0]);");
        engEvalString(global_matlab_engine, "hold on");
        //Plot major axis: 
        std::vector<double> major_axis_x_vec, major_axis_y_vec, second_axis_x_vec, second_axis_y_vec;
        major_axis_x_vec.reserve(matlab_array_size);
        major_axis_y_vec.reserve(matlab_array_size);
        second_axis_x_vec.reserve(matlab_array_size);
        second_axis_y_vec.reserve(matlab_array_size);
        mxArray *major_axis_x_matlabarray, *major_axis_y_matlabarray;
        major_axis_x_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        major_axis_y_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        mxArray *second_axis_x_matlabarray, *second_axis_y_matlabarray;
        second_axis_x_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        second_axis_y_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        const double radius_scaling = sqrt(4.0 *log(2.0));
        for (int i = 0; i < matlab_array_size; i++) {
            auto svd_structure = p_vect[i].covariance_matrix.jacobiSvd(Eigen::ComputeFullU);
            State_variable major_axis = svd_structure.matrixU().col(0) * sqrt(svd_structure.singularValues().operator()(0));
            major_axis_x_vec.push_back(major_axis(first_coor)*radius_scaling);
            major_axis_y_vec.push_back(major_axis(second_coor)*radius_scaling);
            State_variable second_axis = svd_structure.matrixU().col(1) * sqrt(svd_structure.singularValues().operator()(1));
            second_axis_x_vec.push_back(second_axis(first_coor)*radius_scaling);
            second_axis_y_vec.push_back(second_axis(second_coor)*radius_scaling);
        }
        std::copy(major_axis_x_vec.begin(), major_axis_x_vec.end(), mxGetPr(major_axis_x_matlabarray));
        std::copy(major_axis_y_vec.begin(), major_axis_y_vec.end(), mxGetPr(major_axis_y_matlabarray));
        std::copy(second_axis_x_vec.begin(), second_axis_x_vec.end(), mxGetPr(second_axis_x_matlabarray));
        std::copy(second_axis_y_vec.begin(), second_axis_y_vec.end(), mxGetPr(second_axis_y_matlabarray));
        engPutVariable(global_matlab_engine, "major_axis_x", major_axis_x_matlabarray);
        engPutVariable(global_matlab_engine, "major_axis_y", major_axis_y_matlabarray);
        engPutVariable(global_matlab_engine, "second_axis_x", second_axis_x_matlabarray);
        engPutVariable(global_matlab_engine, "second_axis_y", second_axis_y_matlabarray);
        if (!skip_first)
        {
            engEvalString(global_matlab_engine, "pl1=quiver(x,y,major_axis_x,major_axis_y,'k')");
            engEvalString(global_matlab_engine, "pl1.ShowArrowHead='off';pl1.AutoScale='off'");
            engEvalString(global_matlab_engine, "pl1=quiver(x,y,-major_axis_x,-major_axis_y,'k')");
            engEvalString(global_matlab_engine, "pl1.ShowArrowHead='off';pl1.AutoScale='off'");
        }
        if (!skip_second) {
            engEvalString(global_matlab_engine, "pl1=quiver(x,y,second_axis_x,second_axis_y,'g')");
            engEvalString(global_matlab_engine, "pl1.ShowArrowHead='off';pl1.AutoScale='off'");
            engEvalString(global_matlab_engine, "pl1=quiver(x,y,-second_axis_x,-second_axis_y,'g')");
            engEvalString(global_matlab_engine, "pl1.ShowArrowHead='off';pl1.AutoScale='off'");
        }
        engEvalString(global_matlab_engine, "hold off");
        engEvalString(global_matlab_engine, plot_options);
        engEvalString(global_matlab_engine, "drawnow");
        mxDestroyArray(weight_matlabarray);
        mxDestroyArray(x_matlabarray);
        mxDestroyArray(y_matlabarray);
        mxDestroyArray(major_axis_x_matlabarray);
        mxDestroyArray(major_axis_y_matlabarray);
        mxDestroyArray(second_axis_x_matlabarray);
        mxDestroyArray(second_axis_y_matlabarray);
    }
    else if (dimension_count == 3) {
        index_type first_coor, second_coor, third_coor;
        bool first_find = false, second_find = false;
        for (index_type i = 0; i < dimension; i++) {
            if (projection_dimensions[i])
                if (second_find)
                    third_coor = i;
                else if (first_find){
                    second_coor = i;
                    second_find = true;
                }
                else {
                    first_coor = i;
                    first_find = true;
                }
        }
        //Plot position: 
        auto x_vec = get_location_in_dimension_n(first_coor);
        auto y_vec = get_location_in_dimension_n(second_coor);
        auto z_vec = get_location_in_dimension_n(third_coor);
        auto weight_vec = get_weight();
        if (delete_remaining) {
            x_vec.resize(g_plot_max_size_passed);
            y_vec.resize(g_plot_max_size_passed);
            z_vec.resize(g_plot_max_size_passed);
            weight_vec.resize(g_plot_max_size_passed);
        }
        mxArray *x_matlabarray, *y_matlabarray, *z_matlabarray, *weight_matlabarray;
        x_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        std::copy(x_vec.begin(), x_vec.end(), mxGetPr(x_matlabarray));
        y_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        std::copy(y_vec.begin(), y_vec.end(), mxGetPr(y_matlabarray));
        z_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        std::copy(z_vec.begin(), z_vec.end(), mxGetPr(z_matlabarray));
        weight_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        std::copy(weight_vec.begin(), weight_vec.end(), mxGetPr(weight_matlabarray));
        engPutVariable(global_matlab_engine, "x", x_matlabarray);
        engPutVariable(global_matlab_engine, "y", y_matlabarray);
        engPutVariable(global_matlab_engine, "z", z_matlabarray);
        engPutVariable(global_matlab_engine, "w", weight_matlabarray);
        if (overwrite_previous) {
            engEvalString(global_matlab_engine, "cla");
        }
        engEvalString(global_matlab_engine, "scatter3(x,y,z,10000*w,'MarkerEdgeColor',[1 0 0]);");
        engEvalString(global_matlab_engine, "hold on");
        //Plot major axis: 
        std::vector<double> major_axis_x_vec, major_axis_y_vec, major_axis_z_vec, second_axis_x_vec, second_axis_y_vec, second_axis_z_vec;
        major_axis_x_vec.reserve(matlab_array_size);
        major_axis_y_vec.reserve(matlab_array_size);
        major_axis_z_vec.reserve(matlab_array_size);
        second_axis_x_vec.reserve(matlab_array_size);
        second_axis_y_vec.reserve(matlab_array_size);
        second_axis_z_vec.reserve(matlab_array_size);
        mxArray *major_axis_x_matlabarray, *major_axis_y_matlabarray, *major_axis_z_matlabarray;
        major_axis_x_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        major_axis_y_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        major_axis_z_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        mxArray *second_axis_x_matlabarray, *second_axis_y_matlabarray, *second_axis_z_matlabarray;
        second_axis_x_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        second_axis_y_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        second_axis_z_matlabarray = mxCreateDoubleMatrix(1, matlab_array_size, mxREAL);
        const double radius_scaling = sqrt(4.0 *log(2.0));
        for (int i = 0; i < matlab_array_size; i++) {
            auto svd_structure = p_vect[i].covariance_matrix.jacobiSvd(Eigen::ComputeFullU);
            State_variable major_axis = svd_structure.matrixU().col(0) * sqrt(svd_structure.singularValues().operator()(0));
            major_axis_x_vec.push_back(major_axis(first_coor)*radius_scaling);
            major_axis_y_vec.push_back(major_axis(second_coor)*radius_scaling);
            major_axis_z_vec.push_back(major_axis(third_coor)*radius_scaling);
            State_variable second_axis = svd_structure.matrixU().col(1) * sqrt(svd_structure.singularValues().operator()(1));
            second_axis_x_vec.push_back(second_axis(first_coor)*radius_scaling);
            second_axis_y_vec.push_back(second_axis(second_coor)*radius_scaling);
            second_axis_z_vec.push_back(second_axis(third_coor)*radius_scaling);
        }
        std::copy(major_axis_x_vec.begin(), major_axis_x_vec.end(), mxGetPr(major_axis_x_matlabarray));
        std::copy(major_axis_y_vec.begin(), major_axis_y_vec.end(), mxGetPr(major_axis_y_matlabarray));
        std::copy(major_axis_z_vec.begin(), major_axis_z_vec.end(), mxGetPr(major_axis_z_matlabarray));
        std::copy(second_axis_x_vec.begin(), second_axis_x_vec.end(), mxGetPr(second_axis_x_matlabarray));
        std::copy(second_axis_y_vec.begin(), second_axis_y_vec.end(), mxGetPr(second_axis_y_matlabarray));
        std::copy(second_axis_z_vec.begin(), second_axis_z_vec.end(), mxGetPr(second_axis_z_matlabarray));
        engPutVariable(global_matlab_engine, "major_axis_x", major_axis_x_matlabarray);
        engPutVariable(global_matlab_engine, "major_axis_y", major_axis_y_matlabarray);
        engPutVariable(global_matlab_engine, "major_axis_z", major_axis_z_matlabarray);
        engPutVariable(global_matlab_engine, "second_axis_x", second_axis_x_matlabarray);
        engPutVariable(global_matlab_engine, "second_axis_y", second_axis_y_matlabarray);
        engPutVariable(global_matlab_engine, "second_axis_z", second_axis_z_matlabarray);
        if (!skip_first)
        {
            engEvalString(global_matlab_engine, "pl1=quiver3(x,y,z,major_axis_x,major_axis_y,major_axis_z,'k')");
            engEvalString(global_matlab_engine, "pl1.ShowArrowHead='off';pl1.AutoScale='off'");
            engEvalString(global_matlab_engine, "pl1=quiver3(x,y,z,-major_axis_x,-major_axis_y,-major_axis_z,'k')");
            engEvalString(global_matlab_engine, "pl1.ShowArrowHead='off';pl1.AutoScale='off'");
        }
        if (!skip_second) {
            engEvalString(global_matlab_engine, "pl1=quiver3(x,y,z,second_axis_x,second_axis_y,second_axis_z,'g')");
            engEvalString(global_matlab_engine, "pl1.ShowArrowHead='off';pl1.AutoScale='off'");
            engEvalString(global_matlab_engine, "pl1=quiver3(x,y,z,-second_axis_x,-second_axis_y,-second_axis_z,'g')");
            engEvalString(global_matlab_engine, "pl1.ShowArrowHead='off';pl1.AutoScale='off'");
        }
        engEvalString(global_matlab_engine, "hold off");
        engEvalString(global_matlab_engine, plot_options);
        engEvalString(global_matlab_engine, "drawnow");
        mxDestroyArray(weight_matlabarray);
        mxDestroyArray(x_matlabarray);
        mxDestroyArray(y_matlabarray);
        mxDestroyArray(z_matlabarray);
        mxDestroyArray(major_axis_x_matlabarray);
        mxDestroyArray(major_axis_y_matlabarray);
        mxDestroyArray(major_axis_z_matlabarray);
        mxDestroyArray(second_axis_x_matlabarray);
        mxDestroyArray(second_axis_y_matlabarray);
        mxDestroyArray(second_axis_z_matlabarray);
    }
    else {
        std::cout << "Can only plot for 2 or 3 dimensions, the dimensions input is not valid.\n";
    }
    return global_matlab_engine;
}
Plot_handle Population_density::plot(const char* plot_options, const int plot_flags){
    std::vector<bool> range_dimensions(dimension, false);
    range_dimensions[0] = range_dimensions[1] = true;
    return plot(range_dimensions, plot_options, plot_flags);
}
Plot_handle Population_density::output_center_and_weight() const{
    auto weight_vec = get_weight();
    mxArray *weight_matlabarray;
    weight_matlabarray = mxCreateDoubleMatrix(1, weight_vec.size(), mxREAL);
    std::copy(weight_vec.begin(), weight_vec.end(), mxGetPr(weight_matlabarray));
    engPutVariable(global_matlab_engine, "w", weight_matlabarray);
    mxArray *x_matlabarray;
    char line[255];
    sprintf(line, "X = zeros(%u,%d);", p_vect.size(), dimension);
    engEvalString(global_matlab_engine, line);
    for (int i = 0; i < dimension; i++) {
        auto x_vec = get_location_in_dimension_n(i);
        x_matlabarray = mxCreateDoubleMatrix(1, x_vec.size(), mxREAL);
        std::copy(x_vec.begin(), x_vec.end(), mxGetPr(x_matlabarray));
        engPutVariable(global_matlab_engine, "x", x_matlabarray);
        sprintf(line, "X(:,%d)=x;", i + 1);
        engEvalString(global_matlab_engine, line);
    }

    return global_matlab_engine;
}
Plot_handle Population_density::copy_particle_to_matlab(const index_type i) const {
    mxArray *x_matlabarray;
    mxArray *sigma_matlabarray;
    x_matlabarray = mxCreateDoubleMatrix(1, dimension, mxREAL);
    sigma_matlabarray = mxCreateDoubleMatrix(dimension, dimension, mxREAL);
    memcpy(mxGetPr(x_matlabarray), p_vect[i].center_location.data(), dimension * sizeof(double));
    memcpy(mxGetPr(sigma_matlabarray), p_vect[i].covariance_matrix.data(), dimension*dimension * sizeof(double));
    //std::copy(p_vect[i].center_location.data(), p_vect[i].center_location.data() + p_vect[i].center_location.size(), x_matlabarray);//
    //std::copy(p_vect[i].covariance_matrix.data(), p_vect[i].covariance_matrix.data() + p_vect[i].covariance_matrix.size(), sigma_matlabarray);
    //Copy order does not matter since the matrix is self-adjoint.
    engPutVariable(global_matlab_engine, "x_single", x_matlabarray);
    engPutVariable(global_matlab_engine, "sigma_single", sigma_matlabarray);
    mxDestroyArray(x_matlabarray);
    mxDestroyArray(sigma_matlabarray);
    return global_matlab_engine;
}
void Population_density::output_particle(const index_type i, const char output_filename[]) const {
    MATFile* pmat;
    pmat = matOpen(output_filename, "w");
    if (pmat == NULL) {
        std::cout << "Cannot write file.\n";
        return;
    }
    mxArray* w_matlabarray;
    mxArray* x_matlabarray;
    mxArray* sigma_matlabarray;
    size_t sigma_dimensions[3];
    sigma_dimensions[0] = sigma_dimensions[1] = dimension;
    sigma_dimensions[2] = 1;
    sigma_matlabarray = mxCreateNumericArray(3, sigma_dimensions, mxDOUBLE_CLASS, mxREAL);
    w_matlabarray = mxCreateDoubleMatrix(1, 1, mxREAL);
    x_matlabarray = mxCreateDoubleMatrix(dimension, 1, mxREAL);
    memcpy(mxGetPr(w_matlabarray), &p_vect[i].weight, sizeof(double));
    memcpy(mxGetPr(x_matlabarray), p_vect[i].center_location.data(), dimension * sizeof(double));
    memcpy(mxGetPr(sigma_matlabarray), p_vect[i].covariance_matrix.data(), dimension * dimension * sizeof(double));

    //std::copy(p_vect[i].center_location.data(), p_vect[i].center_location.data() + p_vect[i].center_location.size(), x_matlabarray);//
    //std::copy(p_vect[i].covariance_matrix.data(), p_vect[i].covariance_matrix.data() + p_vect[i].covariance_matrix.size(), sigma_matlabarray);
    //Copy order does not matter since the matrix is self-adjoint.
    matPutVariable(pmat, "w_array", w_matlabarray);
    matPutVariable(pmat, "x_array", x_matlabarray);
    matPutVariable(pmat, "sigma_array", sigma_matlabarray);
    mxDestroyArray(w_matlabarray);
    mxDestroyArray(x_matlabarray);
    mxDestroyArray(sigma_matlabarray);
    matClose(pmat);
    return;
}
#endif

#ifdef MATLAB_DATAIO
void Population_density::output_all_particles(const char output_filename[]) const {
    MATFile *pmat;
    pmat = matOpen(output_filename, "w");
    if (pmat == NULL) {
        std::cout << "Cannot write file.\n";
        return;
    }
    mxArray *w_matlabarray;
    mxArray *x_matlabarray;
    mxArray *sigma_matlabarray;
    size_t sigma_dimensions[3];
    sigma_dimensions[0] = sigma_dimensions[1] = dimension;
    sigma_dimensions[2] = size();
    sigma_matlabarray = mxCreateNumericArray(3, sigma_dimensions, mxDOUBLE_CLASS, mxREAL);
    w_matlabarray = mxCreateDoubleMatrix(1,size(), mxREAL);
    x_matlabarray = mxCreateDoubleMatrix(dimension, size(), mxREAL);
    for (int i = 0; i < size(); i++) {
        memcpy(mxGetPr(w_matlabarray)+i, &p_vect[i].weight, sizeof(double));
        memcpy(mxGetPr(x_matlabarray)+i*dimension, p_vect[i].center_location.data(), dimension * sizeof(double));
        memcpy(mxGetPr(sigma_matlabarray)+i*dimension*dimension, p_vect[i].covariance_matrix.data(), dimension*dimension * sizeof(double));
    }

    //std::copy(p_vect[i].center_location.data(), p_vect[i].center_location.data() + p_vect[i].center_location.size(), x_matlabarray);//
    //std::copy(p_vect[i].covariance_matrix.data(), p_vect[i].covariance_matrix.data() + p_vect[i].covariance_matrix.size(), sigma_matlabarray);
    //Copy order does not matter since the matrix is self-adjoint.
    matPutVariable(pmat, "w_array", w_matlabarray);
    matPutVariable(pmat, "x_array", x_matlabarray);
    matPutVariable(pmat, "sigma_array", sigma_matlabarray);
    mxDestroyArray(w_matlabarray);
    mxDestroyArray(x_matlabarray);
    mxDestroyArray(sigma_matlabarray);
    matClose(pmat);
    return;
}

void Population_density::input_all_particles(const char input_filename[]){
    MATFile *pmat;
    pmat = matOpen(input_filename, "r");
    int input_particle_count;
    //Check validity of file
    if (pmat == NULL) {
        std::cout << "Fail to open mat file.\n";
        return;
    }
    mxArray *matlabinfoarray;
    matlabinfoarray = matGetVariableInfo(pmat, "w_array");
    if (matlabinfoarray == NULL) {
        std::cout << "mat file does not contain w_array.\n";
        return;
    }
    if (!mxGetNumberOfDimensions(matlabinfoarray) == 2) {
        std::cout << "w_array is not of dimension 2\n";
        return;
    }
    input_particle_count = mxGetNumberOfElements(matlabinfoarray);
    matlabinfoarray = matGetVariableInfo(pmat, "x_array");
    if (matlabinfoarray == NULL) {
        std::cout << "mat file does not contain x_array.\n";
        return;
    }
    if (mxGetNumberOfDimensions(matlabinfoarray) != 2 || mxGetNumberOfElements(matlabinfoarray) != input_particle_count*dimension) {
        std::cout << "x_array is not of right size.\n";
        return;
    }
    matlabinfoarray = matGetVariableInfo(pmat, "sigma_array");
    if (matlabinfoarray == NULL) {
        std::cout << "mat file does not contain sigma_array.\n";
        return;
    }
    if (mxGetNumberOfDimensions(matlabinfoarray) != 3 && input_particle_count != 1 || mxGetNumberOfElements(matlabinfoarray) != input_particle_count*dimension*dimension) {
        //For a single particle, the simga_array is 1 dimensional.
        std::cout << "sigma_array is not of right size.\n";
        std::cout << "Expect " << input_particle_count << " * "<< dimension << "^2, but input is of size " << mxGetNumberOfElements(matlabinfoarray);
        return;
    }
    mxDestroyArray(matlabinfoarray);
    matClose(pmat);
    //Finished checking. Reading data:
    mxArray *w_matlabarray;
    mxArray *x_matlabarray;
    mxArray *sigma_matlabarray;
    pmat = matOpen(input_filename, "r");
    w_matlabarray = matGetVariable(pmat, "w_array");
    x_matlabarray = matGetVariable(pmat, "x_array");
    sigma_matlabarray = matGetVariable(pmat, "sigma_array");
    const auto w_eigenvector = Eigen::Map<Eigen::VectorXd>(mxGetPr(w_matlabarray), input_particle_count);
    const auto x_eigenmatrix = Eigen::Map<Eigen::MatrixXd>(mxGetPr(x_matlabarray), dimension, input_particle_count);
    auto sigmalinear_eigenmatrix = Eigen::Map<Eigen::MatrixXd>(mxGetPr(sigma_matlabarray), dimension*dimension, input_particle_count);
    p_vect.resize(input_particle_count);
    Particle input_particle(dimension);
    for (int particle_index = 0; particle_index < input_particle_count; particle_index++) {
        input_particle.weight = w_eigenvector[particle_index];
        input_particle.center_location = x_eigenmatrix.col(particle_index);
        const Eigen::Map<Eigen::MatrixXd> sigma_matrix(sigmalinear_eigenmatrix.col(particle_index).data(), dimension, dimension);
        input_particle.covariance_matrix = sigma_matrix;
        p_vect[particle_index] = input_particle;
    }
    mxDestroyArray(w_matlabarray);
    mxDestroyArray(x_matlabarray);
    mxDestroyArray(sigma_matlabarray);
    matClose(pmat);
    return;
}
#else
void Population_density::output_all_particles(const char output_filename[]) const {
    /*
    //Copy data from individual particle to different order:
    std::vector<double> w_vec(p_vect.size());
    std::vector<double> x_vec(p_vect.size() * dimension);
    std::vector<double> sigma_vec(p_vect.size() * dimension * dimension);
    for (int i = 0; i < p_vect.size(); i++) {
        w_vec[i] = p_vect[i].weight;
        std::copy_n(p_vect[i].center_location.data(),dimension, &x_vec[i * dimension]);
        //Warning: this line works because covariance is symmetric
        std::copy_n(p_vect[i].covariance_matrix.data(), dimension * dimension, &sigma_vec[i * dimension * dimension]);
    }
    try {
        H5std_string filename_str = output_filename;
        H5::H5File file(filename_str, H5F_ACC_TRUNC);
        H5::DataSet w_h5array;//particle_count
        H5::DataSet x_h5array;//particle_count  * dimension
        H5::DataSet sigma_h5array;//particle_count * dimension * dimension 
        hsize_t w_dims[2] = { p_vect.size(),1 };
        hsize_t x_dims[2] = { p_vect.size(),dimension };
        hsize_t sigma_dims[3] = { p_vect.size(),dimension,dimension };
        H5::DataSpace w_dataspace(2, w_dims);
        H5::DataSpace x_dataspace(2, x_dims);
        H5::DataSpace sigma_dataspace(3, sigma_dims);
        w_h5array = file.createDataSet("/w_array", H5::PredType::NATIVE_DOUBLE,w_dataspace);
        x_h5array = file.createDataSet("/x_array", H5::PredType::NATIVE_DOUBLE, x_dataspace);
        sigma_h5array = file.createDataSet("/sigma_array", H5::PredType::NATIVE_DOUBLE, sigma_dataspace);
        w_h5array.write(w_vec.data(), H5::PredType::NATIVE_DOUBLE);
        x_h5array.write(x_vec.data(), H5::PredType::NATIVE_DOUBLE);
        sigma_h5array.write(sigma_vec.data(), H5::PredType::NATIVE_DOUBLE);
        w_h5array.close();
        x_h5array.close();
        sigma_h5array.close();
        w_dataspace.close();
        x_dataspace.close();
        sigma_dataspace.close();
        file.close();
    }
    catch (H5::FileIException error)
    {
        error.printErrorStack();
        return;
    }
    // catch failure caused by the DataSet operations
    catch (H5::DataSetIException error)
    {
        error.printErrorStack();
        return;
    }
    // catch failure caused by the DataSpace operations
    catch (H5::DataSpaceIException error)
    {
        error.printErrorStack();
        return;
    }
    // catch failure caused by the DataSpace operations
    catch (H5::DataTypeIException error)
    {
        error.printErrorStack();
        return;
    }
    return;
    */
}
void Population_density::input_all_particles(const char input_filename[]) {
    /*
    H5std_string filename_str = input_filename;
    //NOTE: Since Eigen library uses column major while H5 uses row major, the matrices
    //stored are in transpose.
    H5::DataSet w_h5array;//particle_count
    H5::DataSet x_h5array;//particle_count  * dimension
    H5::DataSet sigma_h5array;//particle_count * dimension * dimension 
    //(Note order of last 2 dimension does not matter since sigma matrix is symmetric.)
    try {
        H5::H5File file(filename_str, H5F_ACC_RDONLY);
        w_h5array = file.openDataSet("/w_array");
        x_h5array = file.openDataSet("/x_array");
        sigma_h5array = file.openDataSet("/sigma_array");
        H5::FloatType ftype = w_h5array.getFloatType();
        H5::DataSpace w_dataspace = w_h5array.getSpace();
        H5::DataSpace x_dataspace = x_h5array.getSpace();
        H5::DataSpace sigma_dataspace = sigma_h5array.getSpace();
        hsize_t w_dims[2];
        w_dataspace.getSimpleExtentDims(w_dims);
        const int input_particle_count = w_dims[0] * w_dims[1];
        hsize_t x_dims[2];
        x_dataspace.getSimpleExtentDims(x_dims);
        if (x_dims[0] != input_particle_count || x_dims[1] != dimension) {
            std::cout << "Dimension of x_array does not match.\n";
            std::cout << x_dims[0] << "," << x_dims[1] << std::endl;
            std::cout << input_particle_count << "," << dimension << std::endl;
            return;
        }
        hsize_t sigma_dims[3];
        sigma_dataspace.getSimpleExtentDims(sigma_dims);
        if (sigma_dims[0] != input_particle_count || sigma_dims[1] != dimension || sigma_dims[2] != dimension) {
            std::cout << "Dimension of sigma_array does not match.\n";
            return;
        }
        Eigen::VectorXd w_eigenvector(input_particle_count);
        w_h5array.read(w_eigenvector.data(), ftype);
        Eigen::MatrixXd x_eigenmatrix(dimension, input_particle_count);
        x_h5array.read(x_eigenmatrix.data(), ftype);
        Eigen::MatrixXd sigmalinear_eigenmatrix(dimension * dimension, input_particle_count);
        sigma_h5array.read(sigmalinear_eigenmatrix.data(), ftype);
        p_vect.resize(input_particle_count);
        Particle input_particle(dimension);
        for (int particle_index = 0; particle_index < input_particle_count; particle_index++) {
            input_particle.weight = w_eigenvector[particle_index];
            input_particle.center_location = x_eigenmatrix.col(particle_index);
            const Eigen::Map<Eigen::MatrixXd> sigma_matrix(sigmalinear_eigenmatrix.col(particle_index).data(), dimension, dimension);
            input_particle.covariance_matrix = sigma_matrix;
            p_vect[particle_index] = input_particle;
        }
    }
    catch (H5::FileIException error)
    {
        error.printErrorStack();
        return;
    }
    // catch failure caused by the DataSet operations
    catch (H5::DataSetIException error)
    {
        error.printErrorStack();
        return;
    }
    // catch failure caused by the DataSpace operations
    catch (H5::DataSpaceIException error)
    {
        error.printErrorStack();
        return;
    }
    // catch failure caused by the DataSpace operations
    catch (H5::DataTypeIException error)
    {
        error.printErrorStack();
        return;
    }
    return;
    */
}
#endif //MATLAB_DATAIO