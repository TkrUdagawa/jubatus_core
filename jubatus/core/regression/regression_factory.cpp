// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2012 Preferred Networks and Nippon Telegraph and Telephone Corporation.
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

#include "regression_factory.hpp"

#include <stdexcept>
#include <string>

#include "regression.hpp"
#include "../common/exception.hpp"
#include "../common/jsonconfig.hpp"

using jubatus::core::common::jsonconfig::config_cast_check;
using jubatus::util::lang::shared_ptr;

namespace jubatus {
namespace core {
namespace regression {

shared_ptr<regression_base> regression_factory::create_regression(
    const std::string& name,
    const common::jsonconfig::config& param,
    shared_ptr<storage::storage_base> storage) {
  if (name == "PA" || name == "passive_aggressive") {
    return shared_ptr<regression_base>(new regression::passive_aggressive(
      config_cast_check<regression::passive_aggressive::config>(param),
      storage));
  } else if (name == "perceptron") {
    return shared_ptr<regression_base>(new regression::perceptron(
      config_cast_check<regression::perceptron::config>(param),
      storage));
  } else if (name == "CW" || name == "confidence_weighted") {
    return shared_ptr<regression_base>(new regression::confidence_weighted(
      config_cast_check<regression::confidence_weighted::config>(param), 
      storage));
  } else if (name == "AROW") {
    return shared_ptr<regression_base>(new regression::arow(
      config_cast_check<regression::arow::config>(param), 
      storage));
  } else if (name == "NHERD" || name == "normal_herd") {
    return shared_ptr<regression_base>(new regression::normal_herd(
      config_cast_check<regression::normal_herd::config>(param), 
      storage));
  } else {
    throw JUBATUS_EXCEPTION(common::unsupported_method(name));
  }
}

}  // namespace regression
}  // namespace core
}  // namespace jubatus
