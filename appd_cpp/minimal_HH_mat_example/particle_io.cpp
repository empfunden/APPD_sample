#include "particle_method.h"

// Read in initial conditions from a .json file containing lists for centers,
// weights, and covariance matrices.
void Population_density::input_particles(std::string file_name) {
  std::ifstream f(file_name);
  nlohmann::json data = nlohmann::json::parse(f);

  // Read (flattened) particle data into std::vectors.
  int num_particles = data["num_particles"];
  std::vector<double> x_array = data["x_array"];
  std::vector<double> w_array = data["w_array"];
  std::vector<double> m_array = data["m_array"];
  int dim = x_array.size() / w_array.size();

  // Convert the std::vector data into Eigen::Vector/Matrix types
  // while building up the Particle vector.
  p_vect.clear();
  for (int i = 0; i < num_particles; ++i) {
    // Map std::vector data into an Eigen::VectorXd with length dim.
    // .data() gives an iterator to the vector's start; we offset by i*dim.
    State_variable x = Eigen::Map<Eigen::VectorXd>(x_array.data() + i * dim, dim);
    double w = w_array[i];
    // Each vector in m_array has dim^2 numbers; we offset accordingly.
    Matrix_type m = (Eigen::Map<Eigen::VectorXd>(m_array.data() + i * dim * dim, dim * dim))
                    .reshaped(dim, dim);
    Particle p(w, x, m);
    p_vect.push_back(p);
  }
  f.close();
}

// Takes a vector of Particles and outputs their centers, weights, and
// covariance matrices to a .json file.
void Population_density::output_particles(std::string file_name) {
  std::ofstream o(file_name);
  nlohmann::json jsout;  // JSON data structure to store outputed arrays.

  // Return empty file if p_vect has no particles (bad); find dimension.
  if (p_vect.empty()) return;
  int num_particles = p_vect.size();
  int dim = p_vect[0].center_location.rows();
  std::cout << dim << " is dim" << std::endl;

  std::vector<double> x_array(num_particles * dim);
  std::vector<double> w_array(num_particles);
  std::vector<double> m_array(num_particles * dim * dim);

  // Write particle data to vectors (center locations & matrices are flattened).
  for (int i = 0; i < num_particles; ++i) {
    std::copy_n(p_vect[i].center_location.data(), dim, &x_array[i * dim]);
    w_array[i] = p_vect[i].weight;
    // Note that by default, cov_matrix.data() spits out numbers in col, not row
    // order. Since cov_matrix is symmetric, we don't need to worry about this.
    std::copy_n(p_vect[i].covariance_matrix.data(), dim * dim,
                &m_array[i * dim * dim]);
  }

  // Write vectors' data to json struct.
  jsout["num_particles"] = num_particles;
  jsout["x_array"] = x_array;
  jsout["w_array"] = w_array;
  jsout["m_array"] = m_array;

  // Pretty-print json struct to output file.
  o << std::setw(4) << jsout << std::endl;
  o.close();
}
