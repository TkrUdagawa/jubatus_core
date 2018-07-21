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

#ifndef JUBATUS_CORE_CONTEXTUAL_BANDIT_CONTEXTUAL_BANDIT_BASE_HPP_
#define JUBATUS_CORE_CONTEXTUAL_BANDIT_CONTEXTUAL_BANDIT_BASE_HPP_

#include <string>
#include "arm.hpp"
#include "arm_storage.hpp"
#include <msgpack.hpp>
namespace jubatus {
namespace core {

namespace contextual_bandit {

class contextual_bandit_base {
  virtual std::string select_arm(const fv& v) = 0;
  virtual void register_reward(
      const std::string& arm_id,
      const double reward,
      fv& v) = 0;
  virtual void add_arm(const std::string& arm_id) = 0;
  virtual void delete_arm(std::string& arm_id) = 0;
  virtual void mix(const arm_storage::diff_t& lhs, arm_storage::diff_t& rhs);
  virtual void get_diff(arm_storage::diff_t& diff);
  virtual bool put_diff(const arm_storage::diff_t& diff);
  //arm_storage::version get_version();
};

}
}
}

#endif // JUBATUS_CORE_CONTEXTUAL_BANDIT_CONTEXTUAL_BANDIT_BASE_HPP_
