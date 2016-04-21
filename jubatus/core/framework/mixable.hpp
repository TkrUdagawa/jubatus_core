// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2011,2012 Preferred Networks and Nippon Telegraph and Telephone Corporation.
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

#ifndef JUBATUS_CORE_FRAMEWORK_MIXABLE_HPP_
#define JUBATUS_CORE_FRAMEWORK_MIXABLE_HPP_

#include <set>
#include <string>

#include "jubatus/util/lang/function.h"
#include "../common/version.hpp"

namespace jubatus {
namespace core {
namespace framework {

class mixable {
 public:
  typedef jubatus::util::lang::function<void ()> update_callback_t;
  mixable();
  explicit mixable(const std::string& name);
  virtual std::set<std::string> mixables() const;
  virtual ~mixable();

  void set_update_callback(const update_callback_t& callback) {
    update_callback_ = callback;
  }

  virtual void updated() const;

  virtual storage::version get_version() const;
 protected:
  std::set<std::string> mixables_;

 private:
  update_callback_t update_callback_;
};

}  // namespace framework
}  // namespace core
}  // namespace jubatus

#endif  // JUBATUS_CORE_FRAMEWORK_MIXABLE_HPP_
