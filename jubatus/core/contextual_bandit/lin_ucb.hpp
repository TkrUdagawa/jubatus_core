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

#ifndef JUBATUS_CORE_CONTEXTUAL_BANDIT_LIN_UCB_HPP_
#define JUBATUS_CORE_CONTEXTUAL_BANDIT_LIN_UCB_HPP_

#include "contextual_bandit_base.hpp"
#include <msgpack.hpp>
#include "../framework/packer.hpp"
#include "jubatus/util/data/serialization.h"

namespace jubatus {
namespace core {
namespace contextual_bandit {

class lin_ucb : contextual_bandit_base {
 public:
  struct config {
    double alpha;

    template<class Ar>
    void serialize(Ar& ar) {
            ar
              & JUBA_MEMBER(alpha);
    }
  };
  
  explicit lin_ucb(config& conf);
  std::string select_arm(const fv& v);
  void register_reward(
      const std::string& arm_id,
      const double reward,
      fv& v) = 0;
  void add_arm(const std::string& arm_id) = 0;
  void delete_arm(std::string& arm_id) = 0;
  void mix(const arm_storage::diff_t& lhs, arm_storage::diff_t& rhs);
  void get_diff(arm_storage::diff_t& diff);
  bool put_diff(const arm_storage::diff_t& diff);
  void pack(framework::packer& pk) const;
  void unpack(msgpack::object o);
  
 private:
  arm_storage s_;
  double alpha_;
};

}
}
}

#endif // JUBATUS_CORE_CONTEXTUAL_BANDIT_LIN_UCB_HPP_
