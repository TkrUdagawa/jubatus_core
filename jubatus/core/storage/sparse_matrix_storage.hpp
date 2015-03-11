// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2011 Preferred Networks and Nippon Telegraph and Telephone Corporation.
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

#ifndef JUBATUS_CORE_STORAGE_SPARSE_MATRIX_STORAGE_HPP_
#define JUBATUS_CORE_STORAGE_SPARSE_MATRIX_STORAGE_HPP_

#include <string>
#include <utility>
#include <vector>
#include <msgpack.hpp>
#include "jubatus/util/data/unordered_map.h"
#include "jubatus/util/concurrent/mutex.h"
#include "../common/key_manager.hpp"
#include "../common/unordered_map.hpp"
#include "../framework/model.hpp"
#include "storage_type.hpp"

namespace jubatus {
namespace core {
namespace storage {

class sparse_matrix_storage : public framework::model {
 public:
  sparse_matrix_storage();
  ~sparse_matrix_storage();

  sparse_matrix_storage& operator =(const sparse_matrix_storage&);

  void set(const std::string& row, const std::string& column, float val);
  void set_nolock(const std::string& row, const std::string& column, float val);
  void set_row(
      const std::string& row,
      const std::vector<std::pair<std::string, float> >& columns);
  void set_row_nolock(
      const std::string& row,
      const std::vector<std::pair<std::string, float> >& columns);

  float get(const std::string& row, const std::string& column) const;
  float get_nolock(const std::string& row, const std::string& column) const;
  void get_row(
      const std::string& row,
      std::vector<std::pair<std::string, float> >& columns) const;
  void get_row_nolock(
      const std::string& row,
      std::vector<std::pair<std::string, float> >& columns) const;

  float calc_l2norm(const std::string& row) const;
  float calc_l2norm_nolock(const std::string& row) const;
  void remove(const std::string& row, const std::string& column);
  void remove_nolock(const std::string& row, const std::string& column);
  void remove_row(const std::string& row);
  void remove_row_nolock(const std::string& row);
  void get_all_row_ids(std::vector<std::string>& ids) const;
  void clear();

  storage::version get_version() const {
    return storage::version();
  }

  util::concurrent::mutex& get_mutex() const {
    return mutex_;
  }

  void pack(framework::packer& packer) const;
  void unpack(msgpack::object o);

 private:
  mutable util::concurrent::mutex mutex_;
  tbl_t tbl_;
  common::key_manager column2id_;
  storage::version version_;

 public:
  MSGPACK_DEFINE(tbl_, column2id_);
};

}  // namespace storage
}  // namespace core
}  // namespace jubatus

#endif  // JUBATUS_CORE_STORAGE_SPARSE_MATRIX_STORAGE_HPP_
