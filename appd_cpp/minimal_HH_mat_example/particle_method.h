#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#pragma once
#define DEFAULT_SPLIT_REL_ERROR 0.01
const int g_plot_max_size_passed = 2048;
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#endif
//#define DEBUG_DEADLOCK
#include <functional>
#include <new>
#include <vector>
#include <Eigen/Dense>
#include <boost/numeric/odeint.hpp>
#include <boost/operators.hpp>
#include <boost/math/constants/constants.hpp>
#include <tbb/tbb.h>
#include <tbb/concurrent_vector.h>

#include <nlohmann/json.hpp>
#include <fstream>
#include <string>

// #define MATLAB_VISUALIZE//comment this macro if hdf5 version of data input/ouput is desired. 
#ifdef MATLAB_VISUALIZE
#define MATLAB_DATAIO
#endif
#if EIGEN_VERSION_AT_LEAST(3,3,0)//This is required since scalar_add_op is removed from newer versions of eigen
namespace Eigen {
    namespace internal {

        template<typename Scalar>
        struct scalar_add_op {
            // FIXME default copy constructors seems bugged with std::complex<>
            EIGEN_DEVICE_FUNC inline scalar_add_op(const scalar_add_op& other) : m_other(other.m_other) { }
            EIGEN_DEVICE_FUNC inline scalar_add_op(const Scalar& other) : m_other(other) { }
            EIGEN_DEVICE_FUNC inline Scalar operator() (const Scalar& a) const { return a + m_other; }
            template <typename Packet>
            EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const
            {
                return internal::padd(a, pset1<Packet>(m_other));
            }
            const Scalar m_other;
        };
        template<typename Scalar>
        struct functor_traits<scalar_add_op<Scalar> >
        {
            enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = packet_traits<Scalar>::HasAdd };
        };

    } // namespace internal
}
#endif // EIGEN_VERSION_AT_LEAST(3,3,0)
#include <boost/numeric/odeint/external/eigen/eigen.hpp>//Provides support for eigen vectors and matrices as state_type of odeint

typedef Eigen::VectorXd State_variable;
typedef Eigen::MatrixXd Matrix_type;
typedef double value_type;
typedef int index_type;
#ifdef MATLAB_DATAIO
#include "mat.h"
#endif //MATLAB_DATAIO
#ifdef MATLAB_VISUALIZE
#define FAST_PLOT_PARTICLE_COUNT 200
#include "engine.h"
typedef Engine* Plot_handle;
extern Engine* global_matlab_engine;
void passplotcommand(Plot_handle plot_handle, const char* argument);
#endif // MATLAB_VISUALIZE
index_type closest_grid_index(const value_type x, const value_type y, const value_type x_lb, const value_type x_ub, const value_type y_lb, const value_type y_ub, const index_type spatial_resolution);
const value_type pi = boost::math::constants::pi<value_type>();
typedef std::function<State_variable(const State_variable)> vector_vector_function;
typedef std::function<value_type(const State_variable)> vector_scalar_function;
typedef std::function<bool(const State_variable)> vector_bool_function;
typedef std::function<void(State_variable)> vector_void_function;
struct Center_level_set;
struct Advection_diffusion_eqn;
class Population_density;
class Population_density_with_equation;
struct Particle {
public:
    State_variable center_location;
    Matrix_type covariance_matrix;
    value_type weight;
    Particle(index_type state_space_dimension = 1U) {
        weight = 0.0;
        center_location = Eigen::VectorXd::Zero(state_space_dimension);
        covariance_matrix = Eigen::MatrixXd::Identity(state_space_dimension, state_space_dimension);
    }
    Particle(const value_type weight, const State_variable& center_location, const Matrix_type& covariance_matrix) : weight(weight),center_location(center_location),covariance_matrix(covariance_matrix) {}
    value_type density_at(const State_variable& location) const;
    value_type density_projection_at_coordinate(const State_variable& location, const std::vector<bool>& range_dimensions) const;//range_dimensions is of length dimension. e.g. TFTF means projection to dimension 0 and 2
    friend struct Center_level_set;
    friend struct Advection_diffusion_eqn;
    friend class Population_density;
    friend class Population_density_with_equation;
private:
    Center_level_set to_center_level_set() const;
};
//Center_level_set is only used internally.
struct Center_level_set :// "State variable" used in update_ODE to compute advection diffusion equation
    boost::additive1<Center_level_set,
    boost::additive2<Center_level_set, value_type,
    boost::multiplicative2<Center_level_set, value_type>>>
{
    State_variable center;
    Matrix_type spanning_vertices;
    Center_level_set()
        :center(), spanning_vertices() {
    }
    Center_level_set(State_variable v, Matrix_type A)
        :center(v), spanning_vertices(A) {
    }
    Center_level_set& operator+=(const Center_level_set& p) {
        center += p.center;
        spanning_vertices += p.spanning_vertices;
        return *this;
    }
    Center_level_set& operator+=(const value_type a) {
        center = center.array() + a;
        spanning_vertices = spanning_vertices.array() + a;
        return *this;
    }
    Center_level_set& operator*=(const value_type a) {
        center *= a;
        spanning_vertices *= a;
        return *this;
    }
    Center_level_set operator/(const Center_level_set& p2) {
        center = center.cwiseQuotient(p2.center);
        spanning_vertices = spanning_vertices.cwiseQuotient(p2.spanning_vertices);
        return *this;
    }
    Particle to_particle(const value_type weight) const {
        const auto x_current = center;
        const auto W = spanning_vertices;
        const Matrix_type Sigma = W * W.transpose();
        return Particle(weight, x_current, Sigma);
    }
    bool need_split(const Center_level_set&, const Advection_diffusion_eqn&, const value_type rel_error_bound = DEFAULT_SPLIT_REL_ERROR);
};
Center_level_set abs(const Center_level_set& p1);
//Following template defines vector-space algebra on center_level_set, which is required for using odeint solvers.
namespace boost {
    namespace numeric {
        namespace odeint {
            template<>
            struct vector_space_norm_inf<Center_level_set> {
                typedef double result_type;
                double operator()(const Center_level_set& p) const {
                    const value_type left = p.center.cwiseAbs().maxCoeff();
                    const value_type right = p.spanning_vertices.cwiseAbs().maxCoeff();
                    return left>right?left:right;
                }
            };
        }
    }
}
typedef std::function < value_type(const Particle current_state, const Particle prev_state, const value_type delta_t)> coupling_strength_function;
typedef std::function<State_variable(const State_variable state_variable, const value_type coupling_strength)> coupling_velocity_function;
struct Advection_diffusion_eqn {
    //An advection diffusion equation consists of an advection velocity, a matrix-valued diffusion coefficient, a pair of functions to implement coupling, and a pair of functions to check domain.
    vector_vector_function advection_velocity;
    coupling_strength_function coupling_strength;
    coupling_velocity_function coupling_velocity;
    Matrix_type diffusion_coefficient;
    const index_type dimension;
    const bool state_variable_always_valid;//Is true when the state_variable can take any value as long as dimension is correct. Is false when the equation is only valid for a subset of the state space.
    Advection_diffusion_eqn(const index_type dimension, const Matrix_type& general_diffusion_coefficient, const bool state_variable_always_valid) : dimension(dimension), state_variable_always_valid(state_variable_always_valid) {
        diffusion_coefficient = general_diffusion_coefficient;
    }
    Advection_diffusion_eqn(const index_type dimension, const value_type isotropic_diffusion_coefficient, const bool state_variable_always_valid) : dimension(dimension), state_variable_always_valid(state_variable_always_valid) {
        diffusion_coefficient = isotropic_diffusion_coefficient * Eigen::MatrixXd::Identity(dimension, dimension);
    }
    vector_bool_function state_variable_in_domain;//Returns true if the variable is inside the domain.
    vector_vector_function state_variable_restrict_to_domain;//A method that maps particles out of domain back inside. (Such particles can be produced when particles are splitted.)
    //Note: Since a lambda function cannot change value of input
};
typedef tbb::concurrent_vector<Particle> particle_vector;
class Population_density {
public:
    const index_type dimension;
    //private: Note p_vect should be considered private, and is not private only due to limit of tbb::concurrent_vector
    particle_vector p_vect;//container for the particles
public:
    //constructor: 
    Population_density(const index_type state_space_dimension)
        : dimension(state_space_dimension) {}
    typedef particle_vector::iterator iterator;
    typedef particle_vector::const_iterator const_iterator;
    //container interface: 
    Particle& operator[](index_type index) {
        return p_vect[index];
    }
    const Particle& operator[](index_type index) const {
        return p_vect[index];
    }
    iterator begin()
    {
        return p_vect.begin();
    }
    const_iterator begin() const
    {
        return p_vect.begin();
    }
    iterator end()
    {
        return p_vect.end();
    }
    const_iterator end() const
    {
        return p_vect.end();
    }
    size_t size() const
    {
        return p_vect.size();
    }
    void resize(const size_t n)
    {
        p_vect.resize(n);
    }
    particle_vector::iterator append(const Particle& particle) {
        p_vect.push_back(particle);
        //p_vect.insert(end(), particle);
        return end();
    }
    value_type density_at(const State_variable& location) const;//returns population density at location
    value_type density_projection_at_coordinate(const State_variable& location, const std::vector<bool>& range_dimensions) const;//range_dimensions is of length dimension. e.g. TFTF means projection to dimension 0 and 2
    std::vector<value_type> density_at(const std::vector<State_variable>& location) const;//returns population density at all points in the location vector
    std::vector<value_type> density_projection_at_coordinate(const std::vector<State_variable>& location, const std::vector<bool>& range_dimensions) const;//range_dimensions is of length dimension. e.g. TFTF means projection to dimension 0 and 2
    double average_in_index(const int coord_idx) const;//returns the average value for coordinate i. (e.g. in HH model, average_in_index(0) returns average membrane potential.
#ifdef MATLAB_VISUALIZE
    Plot_handle plot_density(std::vector<bool> projection_dimensions, const value_type x_lb, const value_type x_ub, const value_type y_lb, const value_type y_ub, const char* imagesc_options = "") const;
    //Plot the marginal population density projected into the projection_dimensions. 
    //e.g.: in a 4-dimensional space, projection_dimensions=TFTF projects the marginal density into 1st and 3rd dimension. 
    //The projection can be into a 1-dimensional or 2-dimensional subspace 
    //x_lb, x_ub, y_lb, y_ub is the lower and bounds for the coordinates to project to. y_lb, y_ub is not used when projection is into a 1-dimensional subspace
    //imagesc_options passes additional parameters into the imagesc command of matlab.
    Plot_handle plot_density(const value_type x_lb, const value_type x_ub, const value_type y_lb, const value_type y_ub, const char* imagesc_options = "") const;
    //project into first 2 dimensions. 
    Plot_handle plot_density(const int projection_dimension, const value_type x_lb, const value_type x_ub, const char *imagesc_options) const;//project into 1 dimension
    //Plot Flags: 1 overwrite, 2 not plot second eigenvec, 4 not plot first eigenvec.
    Plot_handle plot(std::vector<bool> projection_dimensions, const char* plot_options = "", const int plot_flags = 1);
    Plot_handle plot(const char* plot_options = "", const int plot_flags = 1);//project into first 2 dimensions. 
    Plot_handle output_center_and_weight() const;
    Plot_handle copy_particle_to_matlab(const index_type i) const;//COPY single particle to matlab workspace
    void output_particle(const index_type i, const char output_filename[] = "particle.mat") const;//OUTPUT single particle to a mat file
#endif // MATLAB_VISUALIZE
    void output_all_particles(const char output_filename[] = "particles.mat") const;//OUTPUT particles to a mat or h5 file
    void input_all_particles(const char input_filename[] = "particles.mat");//INPUT particles to a mat or h5 file

    void input_particles(std::string file_name);
    void output_particles(std::string file_name);

    //mat file structure: need to contain the following 3 variables with matching dimensions
    // x_array: dimension*particle_count
    // w_array: 1*particle_count
    // sigma_array: dimension*dimension*particle_count 3_D array
    //hdf5 file structure: //All data is of 64-bit floating-point format
    //need to contain the following 3 datasets with matching particle count and dimension
    // x_array: particle_count*dimension
    // w_array: particle_count*1
    // sigma_array: particle_count*dimension*dimension
protected:
    std::vector<double> get_location_in_dimension_n(const index_type n) const;
    std::vector<double> get_weight() const;
    std::vector<index_type> sort_index_in_nth_coordinate(const index_type coordinate) const;
    void sort_in_nth_coordinate(const index_type coordinate);
    void sort_by_weight();
};
class Population_density_with_equation : public Population_density {
public:
    const Advection_diffusion_eqn adv_diff_eqn;
    //Population_density_with_equation contains advection diffusion equation initialized at declaration
    double coupling_at_previous_timestep() const {
        //returns coupling at previous time step
        return coupling_strength_sum;
    }
    const value_type tau;//tau affect particle combination. Larger tau leads to more aggressive particle combination.
    const value_type lambda;//Tikhonov regulariztion factor
    const value_type alpha;//advection velocity relative distance. alpha small uses level set closer to center.
    const value_type split_relax_weight;//Particles with weight under split_relax_weight will be shrunk instead of be splitted
private:
    void set_ODE(const Advection_diffusion_eqn& adv_diff_eqn);
    value_type coupling_strength_sum = 0.0;
    double rel_error_bound = DEFAULT_SPLIT_REL_ERROR;
    typedef std::function<void(const Center_level_set&, Center_level_set&, const value_type)> adv_diff_eqn_odeint_reformulation;
    adv_diff_eqn_odeint_reformulation adv_diff_eqn_on_levelset;
    adv_diff_eqn_odeint_reformulation adv_diff_eqn_on_levelset_coarse_diffusion;
    vector_vector_function advection_dynamics;
    //std::vector<Center_level_set> center_level_set_vect;
    Particle combine_particle_at_indices(const std::vector<index_type>& indices) const;
    //functions used in update_ODE_adaptive_split: using iterator as input since it is the rval of tbb vector push_back
    value_type update_particle_at_index(const Advection_diffusion_eqn& adv_diff_eqn, const value_type timestep, const value_type coupling_timestep, particle_vector::iterator itr);//returns coupling strength over timestep.
    std::tuple<bool, double, State_variable> update_particle_at_index_single_step(const Advection_diffusion_eqn& adv_diff_eqn, const value_type maximum_timestep, particle_vector::iterator itr);
    //retruns true, 0.0, empty vector if updated to end of timestep. Else, return false, remaining time to update, and axis of Gaussian to split. 
#ifdef DEBUG_DEADLOCK
    std::tuple<bool, double, State_variable> update_particle_at_index_single_step_wrapped(const Advection_diffusion_eqn& adv_diff_eqn, const value_type maximum_timestep, particle_vector::iterator itr);
#endif //DEBUG_DEADLOCK
public:
    //constructor: 
    Population_density_with_equation(const Advection_diffusion_eqn& adv_diff_eqn, const index_type state_space_dimension, const value_type tau = 0.01 / 4.0 / log(2.0), const value_type lambda = 1e-6, const value_type alpha = 0.2, const value_type split_relax_weight = 1e-6)
        : Population_density(state_space_dimension),adv_diff_eqn(adv_diff_eqn), tau(tau), lambda(lambda), alpha(alpha), split_relax_weight(split_relax_weight) {
        set_ODE(adv_diff_eqn);
    }
    void update_ODE_const(const value_type timestep, const index_type stepcount = 1);//Updates population density with fixed timestep
    void update_ODE_adaptive(const value_type timestep, const index_type stepcount = 1);//Update population density with variable timestep that is at most timestep
    void update_ODE_adaptive_split(const value_type coupling_timestep, const index_type stepcount, const value_type rel_error_bound);//Update population density with variable timestep that is at most timestep. Moreover, if particles are too large to be accurate, particles are splitted automatically
    void check_linear_approx(const int particle_index) const;//prints information about accuracy of local linear approximation
    void split_particles(const value_type rel_error_bound = DEFAULT_SPLIT_REL_ERROR);//Split all particles that are too large. No need to call if using update_ODE_adaptive_split
    void combine_particles();//combine particles. Should be called between each update_ODE step
    void uniform_shift(const State_variable& pertubation);//Shift all variables by pertubation
};
//Below are function used internally, or obsolete. They are included in the header only due to unit test. 
void approximate_jacobian(Matrix_type& jacobian, const vector_vector_function& full_derivative, const State_variable& x);//Assumes output is already initialized.
Matrix_type approximate_jacobian(const vector_vector_function& full_derivative, const State_variable& x);
std::vector<Particle> split_particle(const Particle x, const Advection_diffusion_eqn& adv_diff_eqn, const value_type rel_error_bound = DEFAULT_SPLIT_REL_ERROR, const value_type split_relax_weight = 1e-6);
std::vector<Particle> split_particle_old(const Particle x, const value_type tau);//A legacy version to keep compatibility with stuff not actively in use.
bool need_split_in_direction(const Particle& x, const State_variable& offset, const Advection_diffusion_eqn& adv_diff_eqn, const value_type rel_error_bound = DEFAULT_SPLIT_REL_ERROR);
