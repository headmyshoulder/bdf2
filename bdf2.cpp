#include <boost/numeric/odeint/util/bind.hpp>
#include <boost/numeric/odeint/util/unwrap_reference.hpp>
#include <boost/numeric/odeint/stepper/stepper_categories.hpp>

#include <boost/numeric/odeint/util/ublas_wrapper.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resizer.hpp>
#include <boost/numeric/odeint/stepper/implicit_euler.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include <utility>

namespace boost {
namespace numeric {
namespace odeint {

template< class ValueType , class Resizer = initially_resizer >
class bdf2
{
public:

    typedef ValueType time_type;
    typedef ValueType value_type;
    typedef boost::numeric::ublas::vector<value_type> state_type ;
    typedef state_wrapper< state_type > wrapped_state_type;
    typedef state_type deriv_type;
    typedef state_wrapper< deriv_type > wrapped_deriv_type;
    typedef boost::numeric::ublas::matrix< value_type > matrix_type;
    typedef state_wrapper< matrix_type > wrapped_matrix_type;
    typedef boost::numeric::ublas::permutation_matrix< size_t > pmatrix_type;
    typedef state_wrapper< pmatrix_type > wrapped_pmatrix_type;
    typedef Resizer resizer_type;
    typedef stepper_tag stepper_category;
    typedef bdf2< ValueType , Resizer > stepper_type;

    bdf2( value_type epsilon = 1E-6 )
    : m_epsilon( epsilon ) 
    { }
    
    template< typename System >
    void do_step( System system , state_type& x , time_type t , time_type dt )
    {
        typedef typename odeint::unwrap_reference< System >::type system_type;
        typedef typename odeint::unwrap_reference< typename system_type::first_type >::type deriv_func_type;
        typedef typename odeint::unwrap_reference< typename system_type::second_type >::type jacobi_func_type;
        system_type &sys = system;
        deriv_func_type &deriv_func = sys.first;
        jacobi_func_type &jacobi_func = sys.second;   
        m_resizer.adjust_size( x , detail::bind( &stepper_type::template resize_impl<state_type> , detail::ref( *this ) , detail::_1 ) );
         
        state_type x1 = x;// x1 for storing x(k+1)
        // std::cout << x1[0] << " " << x1[1] << "\n";
        implicit_euler< double > imp_euler;
        imp_euler.do_step(system, x1, t, dt);
        //x(k+1) computed from above line for x(k) = 4*x(k+1) - 3*x(k+2) + 2*dt*x'(k+2)
        
        // std::cout << x1[0] << " " << x1[1] << "\n";
        
        
        for( size_t i=0 ; i<x.size() ; ++i )
            m_pm.m_v[i] = i;
        
        t +=dt;// t = t0 + 2*dt for x(k+2) and x'(k+2)
        deriv_func( x1 , m_dxdt.m_v , t ); //m_dxdt.m_v is x'(k+2)
        m_b.m_v = 2.0 * dt * m_dxdt.m_v;
        jacobi_func( x1 , m_jacobi.m_v  , t );
        m_jacobi.m_v *= dt;
        m_jacobi.m_v -= boost::numeric::ublas::identity_matrix< value_type >( x.size() );
        
        solve( m_b.m_v , m_jacobi.m_v );
        m_x.m_v = (4.0 *x1)- (3.0*x) + m_b.m_v;//x(k) after first Newton iteration
        
        std::cout << "Before loop : " << m_x.m_v[0] << " " << m_x.m_v[1] << " " << m_b.m_v[0] << " " << m_b.m_v[1] << "\n";
        
        
        while( boost::numeric::ublas::norm_2( m_b.m_v ) > m_epsilon )
        {
            deriv_func( m_x.m_v , m_dxdt.m_v , t );
            m_b.m_v = -(4.0 *x1)+ (3.0*x) + m_x.m_v + 2.0 * dt * m_dxdt.m_v;
            solve( m_b.m_v , m_jacobi.m_v );
            m_x.m_v -= m_b.m_v;
            
            std::cout << "Inside loop : " << m_x.m_v[0] << " " << m_x.m_v[1] << " " << m_b.m_v[0] << " " << m_b.m_v[1] << "\n";
        }

        x = m_x.m_v;
    } 
private:

    template< class StateIn >
    bool resize_impl( const StateIn &x )
    {
        bool resized = false;
        resized |= adjust_size_by_resizeability( m_dxdt , x , typename is_resizeable<deriv_type>::type() );
        resized |= adjust_size_by_resizeability( m_x , x , typename is_resizeable<state_type>::type() );
        resized |= adjust_size_by_resizeability( m_b , x , typename is_resizeable<deriv_type>::type() );
        resized |= adjust_size_by_resizeability( m_jacobi , x , typename is_resizeable<matrix_type>::type() );
        resized |= adjust_size_by_resizeability( m_pm , x , typename is_resizeable<pmatrix_type>::type() );
        return resized;
    }


    void solve( state_type &x , matrix_type &m )
    {
        int res = boost::numeric::ublas::lu_factorize( m , m_pm.m_v );
        if( res != 0 ) exit(0);
        boost::numeric::ublas::lu_substitute( m , m_pm.m_v , x );
    }

private:

    value_type m_epsilon;
    resizer_type m_resizer;
    wrapped_deriv_type m_dxdt;
    wrapped_state_type m_x;
    wrapped_deriv_type m_b;
    wrapped_matrix_type m_jacobi;
    wrapped_pmatrix_type m_pm;

};
} // odeint
} // numeric
} // boost


typedef boost::numeric::ublas::vector< double > vector_type;
typedef boost::numeric::ublas::matrix< double > matrix_type;

struct stiff_system
{    
    void operator()( const vector_type &x , vector_type &dxdt , double /* t */ )
    {
        dxdt[ 0 ] = -101.0 * x[ 0 ] - 100.0 * x[ 1 ]; //Had there been a coefficient of 't' then dfdt != 0
        dxdt[ 1 ] = x[ 0 ];
    }
};

struct stiff_system_jacobi
{
     void operator()( const vector_type & /* x */ , matrix_type &J , const double & /* t */ )
    {
        J( 0 , 0 ) = -101.0;
        J( 0 , 1 ) = -100.0;
        J( 1 , 0 ) = 1.0;
        J( 1 , 1 ) = 0.0;
    }
};
using namespace std;
int main( int argc , char** argv )
{  
    vector_type x( 2.0, 1.0);
    double t = 0.0;
    double dt = 0.0001;
    boost::numeric::odeint::bdf2<double> solver;
    for( size_t i=0 ; i<100 ; ++i , t+= dt )
    {
        solver.do_step( make_pair( stiff_system() , stiff_system_jacobi() ), x , t , dt );
        cout << t << " " << x[0] << " " << x[1] << "\n\n\n";
    }
    return 0;
}