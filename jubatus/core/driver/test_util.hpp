// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2013 Preferred Infrastructure and Nippon Telegraph and Telephone Corporation.
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

#ifndef JUBATUS_CORE_DRIVER_TEST_UTIL_HPP_
#define JUBATUS_CORE_DRIVER_TEST_UTIL_HPP_

#include <string>
#include <sstream>
#include <vector>

#include "../fv_converter/converter_config.hpp"
#include "../fv_converter/datum_to_fv_converter.hpp"
#include "../framework/mixable.hpp"


jubatus::util::lang::shared_ptr<jubatus::core::fv_converter::datum_to_fv_converter>  // NOLINT
  make_fv_converter() {
  jubatus::util::lang::shared_ptr<jubatus::core::fv_converter::datum_to_fv_converter>  // NOLINT
    converter(
        new jubatus::core::fv_converter::datum_to_fv_converter);

  jubatus::core::fv_converter::string_rule str_rule;
  str_rule.key = "*";
  str_rule.type = "str";
  str_rule.sample_weight = "bin";
  str_rule.global_weight = "bin";
  jubatus::core::fv_converter::num_rule num_rule;
  num_rule.key = "*";
  num_rule.type = "num";

  jubatus::core::fv_converter::converter_config c;
  c.string_rules = std::vector<jubatus::core::fv_converter::string_rule>();
  c.string_rules->push_back(str_rule);
  c.num_rules = std::vector<jubatus::core::fv_converter::num_rule>();
  c.num_rules->push_back(num_rule);

  jubatus::core::fv_converter::initialize_converter(c, *converter);
  return converter;
}

jubatus::util::lang::shared_ptr<jubatus::core::fv_converter::datum_to_fv_converter>  // NOLINT
  make_idf_fv_converter() {
  jubatus::util::lang::shared_ptr<jubatus::core::fv_converter::datum_to_fv_converter>  // NOLINT
    converter(
        new jubatus::core::fv_converter::datum_to_fv_converter);

  jubatus::core::fv_converter::string_rule str_rule;
  str_rule.key = "*";
  str_rule.type = "space";
  str_rule.sample_weight = "bin";
  str_rule.global_weight = "idf";
  jubatus::core::fv_converter::num_rule num_rule;
  num_rule.key = "*";
  num_rule.type = "num";

  jubatus::core::fv_converter::converter_config c;
  c.string_rules = std::vector<jubatus::core::fv_converter::string_rule>();
  c.string_rules->push_back(str_rule);
  c.num_rules = std::vector<jubatus::core::fv_converter::num_rule>();
  c.num_rules->push_back(num_rule);

  jubatus::core::fv_converter::initialize_converter(c, *converter);
  return converter;
}



#endif  // JUBATUS_CORE_DRIVER_TEST_UTIL_HPP_
