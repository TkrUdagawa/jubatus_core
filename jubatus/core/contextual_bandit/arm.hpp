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

#ifndef JUBATUS_CORE_CONTEXTUAL_BANDIT_ARM_HPP_
#define JUBATUS_CORE_CONTEXTUAL_BANDIT_ARM_HPP_

#include <Eigen/Sparse>
#include <jubatus/util/data/unordered_map.h>
#include <jubatus/util/lang/shared_ptr.h>
#include <string>
#include <vector>
#include <utility>
#include <math.h>
#include "../common/unordered_map.hpp"
#include <msgpack.hpp>

namespace msgpack {
template <typename T>
class packer;
}  // namespace msgpack

namespace jubatus {
namespace core {
namespace framework {
class jubatus_packer;
typedef msgpack::packer<jubatus_packer> packer;
}  // namespace framework

namespace contextual_bandit {

typedef std::vector<std::pair<std::string, double> > fv;
typedef jubatus::util::data::unordered_map<std::string, int> id_converter;
typedef jubatus::util::data::unordered_map<std::string, double> weights_diff_t;
typedef std::pair<weights_diff_t, weights_diff_t> arm_diff_t;
// arm_diff: first for weights_vector_diff, second for weights_matrix_diff
typedef jubatus::util::data::unordered_map<std::string, int> id_converter;
typedef jubatus::util::lang::shared_ptr<id_converter> id_ptr;

class arm {
 public:
  arm_diff_t arm_diff_mixed_;
  arm_diff_t arm_diff_unmixed_;
  Eigen::SparseMatrix<double> weights_matrix_;
  Eigen::SparseMatrix<double> weights_matrix_inv_;
  Eigen::SparseMatrix<double> weights_vector_;
  double alpha_;
  id_converter id_converter_;
  
  arm();
  explicit arm(double alpha);
  void resize_matrix(size_t n);
  double calc_ucb(const Eigen::SparseMatrix<double>& sfv);
  bool add_feature(const std::string& feature_name);
  void register_reward(
      const double reward,
      Eigen::SparseMatrix<double>& sfv,
      fv& v,
      std::pair<id_converter, id_converter>& converter);
  void record_diff(
      const fv& v,
      Eigen::SparseMatrix<double>& diff_matrix,
      Eigen::SparseMatrix<double>& diff_vector,
      std::pair<id_converter, id_converter>& converter);
  void get_diff(arm_diff_t& diff);
  bool put_diff(const arm_diff_t& diff, const id_converter& converter);
  void mix(const arm_diff_t& lhs, arm_diff_t& rhs);
  void construct_matrix(const id_converter& converter);
  MSGPACK_DEFINE(alpha_, id_converter_);
};

}  // namespace contextual_bandit
}  // namespace core
}  // namespace jubatus
  
#endif  // JUBATUS_CORE_CONTEXTUAL_BANDIT_ARM_HPP_
