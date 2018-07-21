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

#include "lin_ucb.hpp"
#include "arm.hpp"
#include "../framework/packer.hpp"

using std::string;

namespace jubatus {
namespace core {
namespace contextual_bandit {

lin_ucb::lin_ucb(config& conf) : alpha_(conf.alpha) {
}

  string lin_ucb::select_arm(const fv& v) {
    return s_.get_max_ucb_arm(v);
  }
  void lin_ucb::add_arm(const string& arm_id) {
    s_.add_arm(arm_id);
  }

  void lin_ucb::register_reward(
       const string& arm_id,
       const double reward,
       fv& v) {
    s_.register_reward(arm_id, reward, v);
  }

  void lin_ucb::delete_arm(string& arm_id) {
    s_.delete_arm(arm_id);
  }
  
  void lin_ucb::get_diff(arm_storage::diff_t& diff) {
    s_.get_diff(diff);
  }

  bool lin_ucb::put_diff(const arm_storage::diff_t& diff) {
    s_.put_diff(diff);
    return true;
  }
  
  void lin_ucb::mix(const arm_storage::diff_t& lhs, arm_storage::diff_t& rhs) {
    s_.mix(lhs, rhs);
  }

  void lin_ucb::pack(framework::packer& pk) const {
    pk.pack(s_);
  }
  
  void lin_ucb::unpack(msgpack::object o) {
    o.convert(&s_);
  }

} // namespace contextual_bandit
}
}


