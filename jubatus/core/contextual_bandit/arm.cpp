// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2018 Preferred Networks and Nippon Telegraph and Telephone Corporation.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License version 2.1 as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA


#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <math.h>
#include <algorithm>
#include <sstream>
#include <Eigen/Sparse>
#include <jubatus/util/data/unordered_map.h>
#include "arm.hpp"


using jubatus::util::data::unordered_map;
using std::string;
using std::vector;
using std::pair;
using jubatus::util::lang::shared_ptr;

namespace jubatus {
namespace core {
namespace contextual_bandit {

arm::arm() {
  weights_matrix_.resize(1, 1);
  weights_vector_.resize(1, 1);
  alpha_ = 0.1;
}

arm::arm(double alpha) : alpha_(alpha) {
  weights_matrix_.resize(1, 1);
  weights_vector_.resize(1, 1);
}

Eigen::SparseMatrix<double> calc_inverse(
    const Eigen::SparseMatrix<double> weights_matrix) {
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver;
  if(weights_matrix.nonZeros() > 0) {
    solver.compute(weights_matrix);
    int n = weights_matrix.rows();
    Eigen::SparseMatrix<double> I(n,n);
    I.setIdentity();
    Eigen::SparseMatrix<double> inv = solver.solve(I);
    return inv;
  } else {
    Eigen::SparseMatrix<double> I(1,1);
    return I;
  }

}

double arm::calc_ucb(const Eigen::SparseMatrix<double>& sfv) {
  // validate_feature(v);
  std::cout << "non zeros" << weights_matrix_.nonZeros() << std::endl;
  if (weights_matrix_.nonZeros() > 0) {
    weights_matrix_inv_ = calc_inverse(weights_matrix_);
    Eigen::SparseMatrix<double> theta = weights_matrix_inv_ * weights_vector_;
    // std::cout << "theta" << std::endl;
    Eigen::MatrixXd ucb = sfv.transpose() * weights_matrix_ * sfv;
    // pstd::cout << "ucb" << std::endl;
    Eigen::MatrixXd a = theta.transpose() * sfv;
    // std::cout << "a" << std::endl;
    return a(0,0) + alpha_ * sqrt(ucb(0,0));
  } else {
    return 1;
  }
}

void arm::record_diff(
    const fv& v,
    Eigen::SparseMatrix<double>& diff_matrix,
    Eigen::SparseMatrix<double>& diff_vector,
    std::pair<id_converter, id_converter>& converter) {
  weights_diff_t& weights_vector_diff = arm_diff_unmixed_.first;
  weights_diff_t& weights_matrix_diff = arm_diff_unmixed_.second;
  size_t i, j;
  for(i = 0; i < v.size(); ++i) {
    for(j = i; j < v.size(); ++j) {
      int row, col;
      // search both of mixed and unmixed converters
      if (converter.first.find(v[i].first) != converter.first.end()) {
        row = converter.first[v[i].first];
      } else {
        row = converter.second[v[i].first];
      }

      if (converter.first.find(v[j].first) != converter.first.end()) {
        col = converter.first[v[j].first];
      } else {
        col = converter.second[v[j].first];
      }
      string k = v[i].first + "-" + v[j].first;
      std::cout << "(" << k << "): " << weights_matrix_diff[k] << std::endl;
      weights_matrix_diff[k] = weights_matrix_diff[k] +
        diff_matrix.coeffRef(row, col);
      std::cout << diff_matrix.coeffRef(row, col) << std::endl;
      std::cout << "(" << k << "): " << weights_matrix_diff[k] << std::endl;
    }
    weights_vector_diff[v[i].first] = weights_vector_diff[v[i].first] +
      diff_vector.coeffRef(i, 0);
  }
}

void arm::resize_matrix(size_t n) {
  weights_matrix_.conservativeResize(n, n);
  weights_vector_.conservativeResize(n, 1);
}

void arm::register_reward(
    const double reward,
    Eigen::SparseMatrix<double>& sfv,
    fv& v,
    std::pair<id_converter, id_converter>& converter) {
  jubatus::util::data::unordered_map<std::string, double>::iterator it;
  std::sort(v.begin(), v.end());
  if (sfv.rows() != weights_matrix_.rows()) {
    resize_matrix(sfv.rows());
  }
  for (it = arm_diff_unmixed_.second.begin();
       it != arm_diff_unmixed_.second.end();
       ++it) {
    std::cout << it->first << ":" << it -> second << std::endl;
  }

  Eigen::SparseMatrix<double> diff_matrix = sfv * sfv.transpose();
  Eigen::SparseMatrix<double> diff_vector = reward * sfv;
  record_diff(v, diff_matrix, diff_vector, converter);
  weights_matrix_ = weights_matrix_ + diff_matrix;
  weights_vector_ = weights_vector_ + diff_vector;

  for (it = arm_diff_unmixed_.second.begin();
       it != arm_diff_unmixed_.second.end();
       ++it) {
    std::cout << it->first << ":" << it -> second << std::endl;
  }
  // std::cout << diff_ << std::endl;
  // std::cout << weights_matrix_ << std::endl;
}


void arm::get_diff(arm_diff_t& diff) {
  diff = arm_diff_unmixed_;
}

bool arm::put_diff(const arm_diff_t& diff, const id_converter& converter) {
  {
    // mix vector diff
    weights_diff_t::const_iterator it;
    weights_diff_t& weights_vector_mixed = arm_diff_mixed_.first;
    for(it = diff.first.begin(); it != diff.first.end(); ++it) {
      std::cout << "put diff: " << it->first << std::endl;
      weights_vector_mixed[it->first] = it->second;
      std::cout << "put diff: " << it->first << std::endl; 
    }
  }
  
  {
    // mix matrix diff
    weights_diff_t::const_iterator it;
    weights_diff_t& weights_matrix_mixed = arm_diff_mixed_.second;
    for(it = diff.second.begin(); it != diff.second.end(); ++it) {
      std::cout << "put diff: " << it->first << std::endl; 
      weights_matrix_mixed[it->first] = it->second;
    }
  }
  this->construct_matrix(converter);
  arm_diff_unmixed_.first.clear();
  arm_diff_unmixed_.second.clear();
  return true;
}

std::vector<std::string> split(const std::string& str, char sep)
{
  std::vector<std::string> v;
  std::stringstream ss(str);
  std::string buffer;
  while(std::getline(ss, buffer, sep)) {
    v.push_back(buffer);
  }
  return v;
}


void arm::construct_matrix(const id_converter& converter) {
  weights_diff_t::const_iterator it;
  weights_matrix_.resize(converter.size(), converter.size());
  for(it = arm_diff_unmixed_.second.begin(); it != arm_diff_unmixed_.second.end(); ++it) {
    string sep = "-";
    vector<string> s = split(it->first, sep[0]);
    int row = converter.find(s[0])->second;
    int col = converter.find(s[1])->second;
    weights_matrix_.coeffRef(row, col) = it->second;
  }
}


void arm::mix(const arm_diff_t& lhs, arm_diff_t& rhs) {
  {
    // mix vector diff
    weights_diff_t::const_iterator it;
    for(it = lhs.first.begin(); it != lhs.first.end(); ++it) {
      if(rhs.first.find(it->first) != rhs.first.end()) {
        rhs.first[it->first] += it->second;
      } else {
        rhs.first[it->first] = it->second;
      }
    }
  }
  
  {
    // mix matrix diff
    weights_diff_t::const_iterator it;
    for(it = lhs.second.begin(); it != lhs.second.end(); ++it) {
      if(rhs.second.find(it->first) != rhs.second.end()) {
        rhs.second[it->first] += it->second;
      } else {
        rhs.second[it->first] = it->second;
      }
    }
  }
}

}  // namespace contextual_bandit
}  // namespace core
}  // namespace jubatus
