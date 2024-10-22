#pragma once
//runge_kutta23 method based on
//values from A 3(2) Pair of Runge - Kutta Formulas P. Bogacki, LF Shampine
//code from boost::odeint::stepper::runge_kutta_bogacki_shampine32
/*
[auto_generated]
boost/numeric/odeint/stepper/runge_kutta_bogacki_shampine32.hpp

[begin_description]
Implementation of the Runge Kutta Cash Karp 5(4) method. It uses the generic error stepper.
[end_description]

Copyright 2011-2013 Mario Mulansky
Copyright 2011-2013 Karsten Ahnert

Distributed under the Boost Software License, Version 1.0.
(See accompanying file LICENSE_1_0.txt or
copy at http://www.boost.org/LICENSE_1_0.txt)
*/

#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/container/generation/make_vector.hpp>

#include <boost/numeric/odeint/stepper/explicit_error_generic_rk.hpp>
#include <boost/numeric/odeint/algebra/range_algebra.hpp>
#include <boost/numeric/odeint/algebra/default_operations.hpp>
#include <boost/numeric/odeint/algebra/algebra_dispatcher.hpp>
#include <boost/numeric/odeint/algebra/operations_dispatcher.hpp>

#include <boost/numeric/odeint/util/state_wrapper.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resizer.hpp>

#include <boost/array.hpp>
//Additional include needed: 
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/generation/make_controlled.hpp>


namespace boost {
	namespace numeric {
		namespace odeint {


			template< class Value = double >
			struct rk32_bs_coefficients_a1 : boost::array< Value, 1 >
			{
				rk32_bs_coefficients_a1(void)
				{
					(*this)[0] = static_cast< Value >(1) / static_cast< Value >(2);
				}
			};

			template< class Value = double >
			struct rk32_bs_coefficients_a2 : boost::array< Value, 2 >
			{
				rk32_bs_coefficients_a2(void)
				{
					(*this)[0] = static_cast<Value>(0);
					(*this)[1] = static_cast<Value>(3) / static_cast<Value>(4);
				}
			};


			template< class Value = double >
			struct rk32_bs_coefficients_a3 : boost::array< Value, 3 >
			{
				rk32_bs_coefficients_a3(void)
				{
					(*this)[0] = static_cast<Value>(2) / static_cast<Value>(9);
					(*this)[1] = static_cast<Value>(1) / static_cast<Value>(3);
					(*this)[2] = static_cast<Value>(4) / static_cast<Value>(9);
				}
			};

			template< class Value = double >
			struct rk32_bs_coefficients_b : boost::array< Value, 4 >
			{
				rk32_bs_coefficients_b(void)
				{
					(*this)[0] = static_cast<Value>(2) / static_cast<Value>(9);
					(*this)[1] = static_cast<Value>(1) / static_cast<Value>(3);
					(*this)[2] = static_cast<Value>(4) / static_cast<Value>(9);
					(*this)[3] = static_cast<Value>(0);
				}
			};

			template< class Value = double >
			struct rk32_bs_coefficients_db : boost::array< Value, 4 >
			{
				rk32_bs_coefficients_db(void)
				{
					(*this)[0] = static_cast<Value>(2) / static_cast<Value>(9) - static_cast<Value>(7) / static_cast<Value>(24);
					(*this)[1] = static_cast<Value>(1) / static_cast<Value>(3) - static_cast<Value>(1) / static_cast<Value>(4);
					(*this)[2] = static_cast<Value>(4) / static_cast<Value>(9) - static_cast<Value>(1) / static_cast<Value>(3);
					(*this)[3] = static_cast<Value>(0) - static_cast<Value>(1) / static_cast<Value>(8);
				}
			};


			template< class Value = double >
			struct rk32_bs_coefficients_c : boost::array< Value, 4 >
			{
				rk32_bs_coefficients_c(void)
				{
					(*this)[0] = static_cast<Value>(0);
					(*this)[1] = static_cast<Value>(1) / static_cast<Value>(2);
					(*this)[2] = static_cast<Value>(3) / static_cast<Value>(4);
					(*this)[3] = static_cast<Value>(1);
				}
			};


			template<
				class State,
				class Value = double,
				class Deriv = State,
				class Time = Value,
				class Algebra = typename algebra_dispatcher< State >::algebra_type,
				class Operations = typename operations_dispatcher< State >::operations_type,
				class Resizer = initially_resizer
			>
				class runge_kutta_bogacki_shampine32 : public explicit_error_generic_rk< 4, 3, 3, 2,
				State, Value, Deriv, Time, Algebra, Operations, Resizer >
			{

			public:
				typedef explicit_error_generic_rk< 4, 3, 3, 2, State, Value, Deriv, Time,
					Algebra, Operations, Resizer > stepper_base_type;
				typedef typename stepper_base_type::state_type state_type;
				typedef typename stepper_base_type::value_type value_type;
				typedef typename stepper_base_type::deriv_type deriv_type;
				typedef typename stepper_base_type::time_type time_type;
				typedef typename stepper_base_type::algebra_type algebra_type;
				typedef typename stepper_base_type::operations_type operations_type;
				typedef typename stepper_base_type::resizer_type resizer_typ;

				typedef typename stepper_base_type::stepper_type stepper_type;
				typedef typename stepper_base_type::wrapped_state_type wrapped_state_type;
				typedef typename stepper_base_type::wrapped_deriv_type wrapped_deriv_type;


				runge_kutta_bogacki_shampine32(const algebra_type &algebra = algebra_type()) : stepper_base_type(
					boost::fusion::make_vector(rk32_bs_coefficients_a1<Value>(),
						rk32_bs_coefficients_a2<Value>(),
						rk32_bs_coefficients_a3<Value>()),
					rk32_bs_coefficients_b<Value>(), rk32_bs_coefficients_db<Value>(), rk32_bs_coefficients_c<Value>(),
					algebra)
				{ }
			};
			//Additional template specializations needed for runge_kutta_bogacki_shampine32: 
			template< class State, class Value, class Deriv, class Time, class Algebra, class Operations, class Resize >
			struct get_controller< runge_kutta_bogacki_shampine32< State, Value, Deriv, Time, Algebra, Operations, Resize > >
			{
				typedef runge_kutta_bogacki_shampine32< State, Value, Deriv, Time, Algebra, Operations, Resize > stepper_type;
				typedef controlled_runge_kutta< stepper_type > type;
			};
			/**
			* \class runge_kutta_bogacki_shampine32
			* \brief The Bogacki¨CShampine method.
			*
			* The Runge-Kutta Cash-Karp method is one of the standard methods for
			* solving ordinary differential equations, see
			* <a href="https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method">en.wikipedia.org/wiki/Bogacki¨CShampine_method</a>.
			* The method is explicit and fulfills the Error Stepper concept. Step size control
			* is provided but continuous output is not available for this method.
			*
			* This class derives from explicit_error_stepper_base and inherits its interface via CRTP (current recurring template pattern).
			* Furthermore, it derivs from explicit_error_generic_rk which is a generic Runge-Kutta algorithm with error estimation.
			* For more details see explicit_error_stepper_base and explicit_error_generic_rk.
			*
			* \tparam State The state type.
			* \tparam Value The value type.
			* \tparam Deriv The type representing the time derivative of the state.
			* \tparam Time The time representing the independent variable - the time.
			* \tparam Algebra The algebra type.
			* \tparam Operations The operations type.
			* \tparam Resizer The resizer policy type.
			*/


			/**
			* \fn runge_kutta_bogacki_shampine32::runge_kutta_bogacki_shampine32( const algebra_type &algebra )
			* \brief Constructs the runge_kutta_bogacki_shampine32 class. This constructor can be used as a default
			* constructor if the algebra has a default constructor.
			* \param algebra A copy of algebra is made and stored inside explicit_stepper_base.
			*/
		}
	}
}
