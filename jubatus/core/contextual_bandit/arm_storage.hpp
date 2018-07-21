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

#ifndef JUBATUS_CORE_CONTEXTUAL_BANDIT_ARM_STORAGE_HPP_
#define JUBATUS_CORE_CONTEXTUAL_BANDIT_ARM_STORAGE_HPP_

#include "arm.hpp"
#include <string>
#include <utility>
#include <Eigen/Sparse>
#include <msgpack.hpp>

namespace jubatus {
namespace core {
namespace contextual_bandit {

class arm_storage {
public:
  typedef jubatus::util::data::unordered_map<std::string, int> id_converter;
  typedef jubatus::util::data::unordered_map<std::string, arm> arm_map;
  typedef jubatus::util::data::unordered_map<std::string, arm_diff_t> arm_diffs;
  typedef std::pair<id_converter, arm_diffs> diff_t;

  arm_storage();
  explicit arm_storage(double alpha);
  bool add_arm(const std::string& arm_id);
  std::string get_max_ucb_arm(const fv& v);
  void register_reward(
      const std::string& arm_id,
      const double reward,
      fv& context);
  void make_sfv(const fv& v, Eigen::SparseMatrix<double>& sfv);
  void validate_feature(const fv& v);
  void add_feature(const std::string& feature_name);
  void delete_arm(std::string& arm_id);
  void get_diff(diff_t& diff);
  void put_diff(const diff_t& diff);
  void put_converter_diff(const id_converter& diff);
  void mix(const diff_t& lhs, diff_t& rhs);
  
  MSGPACK_DEFINE(id_converter_diff_, id_converter_, arm_map_, alpha_);

  id_converter id_converter_diff_;
  id_converter id_converter_;

  arm_map arm_map_;
  double alpha_;
};

}  // namespace contextual_bandit
}  // namespace core
}  // namespace jubatus

#endif // JUBATUS_CORE_CONTEXTUAL_BANDIT_ARM_STORAGE_HPP_
