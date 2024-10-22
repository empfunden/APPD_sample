//#define FIRING_THRESHOLD_IGNORE_VARIANCE
struct Model_options {
    value_type diffusion_coeff;
    value_type coupling_strength_coefficient;
    value_type coupling_potential;
};
Advection_diffusion_eqn* set_Hodgkin_Huxley_eqn(const Model_options mo) {
    auto eqn_ptr = new Advection_diffusion_eqn(4, mo.diffusion_coeff, false);
    vector_vector_function advection_dynamics;
    coupling_velocity_function coupling_velocity;
    coupling_strength_function coupling_strength;
    advection_dynamics = [](const State_variable& x) {
        //x[0] = V / 100; rest ordered by M, N, H. 
        const value_type gna = 120;
        const value_type ena = 115; const value_type gk = 36;
        const value_type ek = -12;
        const value_type gl = 0.3;
        const value_type el = 10.613;
        const value_type appcurr = 10.0;
        State_variable dxdt(4U);
        value_type V = x[0] * 100.0;
        V += 1e-6 * (abs(V - 10.0)<5e-7 || abs(V - 25.0)<5e-7 || abs(V - 50.0)<5e-7);
        const value_type M = x[1];
        const value_type N = x[2];
        const value_type H = x[3];
        dxdt[0] = (appcurr + gna * M*M*M*H*(ena - V) + gk * (pow(N, 4))*(ek - V) + gl * (el - V)) / 100.0;
        //function y = Ah(V)
        const value_type Ah = 0.07*exp(-V / 20.0);
        //function y = Am(V)
        const value_type Am = (25.0 - V) / (10.0*(exp((25.0 - V) / 10.0) - 1.0));
        //function y = An(V)
        const value_type An = (10.0 - V) / (100.0*(exp((10.0 - V) / 10.0) - 1.0));
        //function y = Bh(V)
        const value_type Bh = 1.0 / (exp((30.0 - V) / 10.0) + 1.0);
        //function y = Bm(V)
        const value_type Bm = 4.0*exp(-V / 18.0);
        //function y = Bn(V)
        const value_type Bn = 0.125*exp(-V / 80.0);
        //%Bn = 0.125*exp(-V / 19.7); //this is for the Corrected HH model
        dxdt[1] = Am * (1 - M) - Bm * M;
        dxdt[2] = An * (1 - N) - Bn * N;
        dxdt[3] = Ah * (1 - H) - Bh * H;
        //std::cout << dxdt;
        return dxdt;
    };
    const double coupling_strength_coefficient = mo.coupling_strength_coefficient;
    coupling_strength = [coupling_strength_coefficient](const Particle& current_state, const Particle& prev_state, const value_type coupling_time_step) {
        //NOTE: coupling strength should compute the average flow rate over the coupling_time_step period, NOT the instantaneous flow rate at a given time. 
        //The choice is made to avoid computing coupling strength on a derivative, 
        //which would be very sensitive to choice of timestep, and may miss some firing when coupling_time_step is too large. 
        //For a "Threshold type" coupling, this means the value should be divided by coupling_time_step
        const value_type threshold_voltage = 0.45;
        if (current_state.center_location[0] <= prev_state.center_location[0]) {
            //No coupling when the membrane potential is decreasing. 
            return 0.0;
        }

#ifndef FIRING_THRESHOLD_IGNORE_VARIANCE
        //Otherwise, we estimate the flow across the threshold, based on the difference 
        const value_type V_normalized = (current_state.center_location[0] - threshold_voltage) / sqrt(current_state.covariance_matrix(0,0));
        const value_type V_prev_normalized = (prev_state.center_location[0] - threshold_voltage) / sqrt(prev_state.covariance_matrix(0, 0));
        const value_type population_proportion = (erf(V_normalized / sqrt(2)) - erf(V_prev_normalized / sqrt(2)))*0.5;
#else
        //It turns out that this estimation is not as good as just using center: 
        const value_type population_proportion = static_cast<double>(current_state.center_location[0] > threshold_voltage && prev_state.center_location[0] < threshold_voltage);
#endif // !1

        

        return population_proportion * coupling_strength_coefficient / coupling_time_step * current_state.weight;
    };
    const value_type coupling_potential_rescaled = mo.coupling_potential / 100.0;
    coupling_velocity = [coupling_potential_rescaled](const State_variable& target, const value_type coupling_strength) {
        //dV/dt = k(V_c - V) 
        State_variable rval(target.size());
        rval.fill(0.0);
        rval[0] = coupling_strength * (coupling_potential_rescaled - target[0]);
        return rval;
    };
    vector_bool_function state_variable_in_domain;
    state_variable_in_domain = [](const State_variable x) {
        if (x[1]<0.0 || x[1] > 1.0 || x[2] < 0.0 || x[2] > 1.0 || x[3] < 0.0 || x[3] > 1.0)
            return false;
        return true;
    };
    vector_vector_function state_variable_restrict_to_domain;
    state_variable_restrict_to_domain = [](State_variable x) {
        auto x_copy = x;
        if (x_copy[1] < 0.0)
            x_copy[1] = 0.0;
        else if (x_copy[1] > 1.0)
            x_copy[1] = 1.0;
        if (x_copy[2] < 0.0)
            x_copy[2] = 0.0;
        else if (x_copy[2] > 1.0)
            x_copy[2] = 1.0;
        if (x_copy[3] < 0.0)
            x_copy[3] = 0.0;
        else if (x_copy[3] > 1.0)
            x_copy[3] = 1.0;
        return x_copy;
    };
    eqn_ptr->advection_velocity = advection_dynamics;
    eqn_ptr->coupling_strength = coupling_strength;
    eqn_ptr->coupling_velocity = coupling_velocity;
    eqn_ptr->state_variable_in_domain = state_variable_in_domain;
    eqn_ptr->state_variable_restrict_to_domain = state_variable_restrict_to_domain;
    return eqn_ptr;
}