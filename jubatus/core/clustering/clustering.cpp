// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2013 Preferred Networks and Nippon Telegraph and Telephone Corporation.
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

#include "clustering.hpp"

#include <algorithm>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include "jubatus/util/lang/function.h"
#include "jubatus/util/lang/bind.h"
#include "../common/jsonconfig.hpp"
#include "clustering_method_factory.hpp"
#include "storage_factory.hpp"
#include "../framework/mixable.hpp"
#include "storage.hpp"

using jubatus::util::lang::shared_ptr;

namespace jubatus {
namespace core {
namespace clustering {

clustering::clustering(
    shared_ptr<clustering_method> clustering_method,
    shared_ptr<storage> storage) {
  set_clustering_method(clustering_method);
  set_storage(storage);
}

clustering::~clustering() {
}

void clustering::set_storage(shared_ptr<storage> storage) {
  storage->add_event_listener(REVISION_CHANGE,
      jubatus::util::lang::bind(&clustering::update_clusters,
          this, jubatus::util::lang::_1, true));
  storage->add_event_listener(UPDATE,
      jubatus::util::lang::bind(&clustering::update_clusters,
          this, jubatus::util::lang::_1, false));
  storage_.reset(new mixable_storage(storage));
}

jubatus::util::lang::shared_ptr<storage> clustering::get_storage() {
  return storage_->get_model();
}

void clustering::update_clusters(const wplist& points, bool batch) {
  if (batch) {
    clustering_method_->batch_update(points);
  } else {
    clustering_method_->online_update(points);
  }
}

void clustering::set_clustering_method(
    shared_ptr<clustering_method> clustering_method) {
  clustering_method_ = clustering_method;
}

bool clustering::push(const std::vector<weighted_point>& points) {
  jubatus::util::lang::shared_ptr<storage> sto = storage_->get_model();
  for (std::vector<weighted_point>::const_iterator it = points.begin();
       it != points.end(); ++it) {
    sto->add(*it);
  }
  return true;
}

wplist clustering::get_coreset() const {
  return storage_->get_model()->get_all();
}

std::vector<common::sfv_t> clustering::get_k_center() const {
  return clustering_method_->get_k_center();
}

common::sfv_t clustering::get_nearest_center(const common::sfv_t& point) const {
  return clustering_method_->get_nearest_center(point);
}

wplist clustering::get_nearest_members(const common::sfv_t& point) const {
  int64_t clustering_id = clustering_method_->get_nearest_center_index(point);
  if (clustering_id == -1) {
    return wplist();
  }
  return clustering_method_->get_cluster(clustering_id, get_coreset());
}

std::vector<wplist> clustering::get_core_members() const {
  return clustering_method_->get_clusters(get_coreset());
}

size_t clustering::get_revision() const {
  return storage_->get_model()->get_revision();
}

framework::mixable* clustering::get_mixable() const {
  return storage_.get();
}

std::string clustering::type() const {
  return "clustering";
}

void clustering::pack(framework::packer& pk) const {
  storage_->get_model()->pack(pk);
}

void clustering::unpack(msgpack::object o) {
  storage_->get_model()->unpack(o);
}

void clustering::clear() {
  storage_->get_model()->clear();
}

void clustering::do_clustering() {
  clustering_method_->batch_update(storage_->get_model()->get_all());
}

}  // namespace clustering
}  // namespace core
}  // namespace jubatus
