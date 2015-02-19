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

#ifndef JUBATUS_CORE_CLASSIFIER_CLASSIFIER_CONFIG_HPP_
#define JUBATUS_CORE_CLASSIFIER_CLASSIFIER_CONFIG_HPP_

#include "jubatus/util/data/serialization.h"

namespace jubatus {
namespace core {
namespace classifier {

struct classifier_config {
  classifier_config()
      : regularization_weight(1.0f) {
  }

  float regularization_weight;

  template<typename Ar>
  void serialize(Ar& ar) {
    ar & JUBA_NAMED_MEMBER("regularization_weight", regularization_weight);
  }
};

}  // namespace classifier
}  // namespace core
}  // namespace jubatus

#endif  // JUBATUS_CORE_CLASSIFIER_CLASSIFIER_CONFIG_HPP_
