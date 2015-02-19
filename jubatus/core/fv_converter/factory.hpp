// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2014 Preferred Networks and Nippon Telegraph and Telephone Corporation.
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

#ifndef JUBATUS_CORE_FV_CONVERTER_FACTORY_HPP_
#define JUBATUS_CORE_FV_CONVERTER_FACTORY_HPP_

#include <string>
#include "converter_config.hpp"

namespace jubatus {
namespace core {
namespace fv_converter {

class binary_feature;
class combination_feature;
class num_filter;
class num_feature;
class string_feature;
class string_filter;

class factory_extender {
 public:
  virtual ~factory_extender() {}

  virtual binary_feature* create_binary_feature(
      const std::string& name,
      const param_t&) const = 0;
  virtual combination_feature* create_combination_feature(
      const std::string& name,
      const param_t&) const = 0;
  virtual num_filter* create_num_filter(
      const std::string& name,
      const param_t&) const = 0;
  virtual num_feature* create_num_feature(
      const std::string& name,
      const param_t&) const = 0;
  virtual string_feature* create_string_feature(
      const std::string& name,
      const param_t&) const = 0;
  virtual string_filter* create_string_filter(
      const std::string& name,
      const param_t&) const = 0;
};

}  // namespace fv_converter
}  // namespace core
}  // namespace jubatus

#endif  // JUBATUS_CORE_FV_CONVERTER_FACTORY_HPP_
