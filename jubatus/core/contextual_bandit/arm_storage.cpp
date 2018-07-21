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


#include "arm_storage.hpp"
#include <string>
#include <vector>
#include <utility>
#include <iostream>

using std::string;
using jubatus::util::data::unordered_map;
using std::vector;
using std::pair;

namespace jubatus {
namespace core {
namespace contextual_bandit {

arm_storage::arm_storage(double alpha) : alpha_(alpha) {
}

arm_storage::arm_storage() {
  alpha_ = 0.1;
}

bool arm_storage::add_arm(const string& arm_id) {
  if (arm_map_.find(arm_id) == arm_map_.end()) {
    arm_map_[arm_id] = arm(alpha_);
    return true;
  } else {
    return false;
  }
}


string arm_storage::get_max_ucb_arm(const fv& context) {
  size_t dim = id_converter_.size() + id_converter_diff_.size();
  // at least one feature should be learned
  if (dim > 0) {
    arm_map::iterator it;
    Eigen::SparseMatrix<double> sfv(dim, 1);
    make_sfv(context, sfv);
    std::cout << "sfv: " << sfv << std::endl;
    std::cout << "fv: " << context[0].first << context[0].second << std::endl;
    string max_arm_id;
    double max_ucb_score = 0.0;
    for (it = arm_map_.begin(); it != arm_map_.end(); ++it) {
      double ucb_score = it->second.calc_ucb(sfv);
      std::cout << "arm: " << it->first << " ucb score: " << ucb_score << std::endl;
      if (ucb_score > max_ucb_score) {
        max_ucb_score = ucb_score;
        max_arm_id = it->first;

      }
    }
    std::cout << "max_ucb_score" << max_ucb_score << std::endl;
    return max_arm_id;
  } else {
    return arm_map_.begin()->first;
  }
}


void arm_storage::validate_feature(const fv& v) {
  fv::const_iterator it;
  for (it = v.begin(); it != v.end(); ++it) {
    unordered_map<string, int>::iterator jt;
    if(id_converter_.find(it->first) == id_converter_.end() &&
       id_converter_diff_.find(it->first) == id_converter_diff_.end()) {
      add_feature(it->first);
    }
  }
}


void arm_storage::add_feature(const string& feature_name) {
  size_t idx = id_converter_.size() + id_converter_diff_.size();
  std::cout << idx << std::endl;
  id_converter_diff_[feature_name] = idx;
}


void arm_storage::register_reward(
    const string& arm_id,
    const double reward,
    fv& context) {
  validate_feature(context);
  std::cout << "validate_feature" << std::endl;
  size_t sfv_size = id_converter_.size() + id_converter_diff_.size();
  Eigen::SparseMatrix<double> sfv(sfv_size, 1);
  make_sfv(context, sfv);
  pair<id_converter, id_converter> conv(id_converter_, id_converter_diff_);
  arm_map_[arm_id].register_reward(
      reward,
      sfv,
      context,
      conv);
}


void arm_storage::make_sfv(const fv& v, Eigen::SparseMatrix<double>& sfv) {
  vector<pair<string, double> >::const_iterator iter;
  for(iter = v.begin(); iter != v.end(); ++iter) {
    if (id_converter_diff_.find(iter->first) != id_converter_diff_.end()) {
      sfv.insert(id_converter_diff_[iter->first], 0) = iter->second;
    } else {
      sfv.insert(id_converter_[iter->first], 0) = iter->second;
    }
  }
}


void arm_storage::get_diff(diff_t& diff) {
  diff.first = id_converter_diff_;
  arm_map::iterator it;
  for (it = arm_map_.begin(); it != arm_map_.end(); ++it) {
    it->second.get_diff(diff.second[it->first]);
  }
}


void arm_storage::put_converter_diff(const id_converter& diff) {
  id_converter::const_iterator it;
  for(it = diff.begin(); it != diff.end(); ++it) {
    id_converter_[it->first] = it->second;
  }
  id_converter_diff_.clear();
}


void arm_storage::put_diff(const diff_t& diff) {
  // get diff from mix master and replace my diff
  // with it.
  put_converter_diff(diff.first);
  arm_diffs::const_iterator it;
  for(it = diff.second.begin(); it != diff.second.end(); ++it) {
    arm_map_[it->first].put_diff(it->second, id_converter_);
  }
}

void arm_storage::delete_arm(string& arm_id) {
  arm_map_.erase(arm_id);
}
  
void arm_storage::mix(const diff_t& lhs, diff_t& rhs) {
  id_converter::const_iterator it;
  
  id_converter& id_master = rhs.first;
  for(it = lhs.first.begin(); it != lhs.first.end(); ++it) {
    if(id_master.find(it->first) == id_master.end()) {
      id_master[it->first] = id_converter_.size() + id_master.size();
    }
  }

  arm_diffs::const_iterator jt;
  for(jt = lhs.second.begin(); jt != lhs.second.end(); ++jt) {
    arm_map_[jt->first].mix(jt->second, rhs.second[jt->first]);
  }
}

}  // namespace contextual_bandit
}  // namespace core
}  // namespace jubatus
