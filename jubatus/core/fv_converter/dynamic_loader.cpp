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

#include <dlfcn.h>
#include <stdlib.h>

#include <iostream>
#include <string>

#include "dynamic_loader.hpp"
#include "exception.hpp"

namespace jubatus {
namespace core {
namespace fv_converter {

dynamic_loader::dynamic_loader(const std::string& path)
    : handle_(0) {

  // Load the plugin with the given path
  handle_ = ::dlopen(path.c_str(), RTLD_LAZY);

  if (!handle_) {
    char* error = dlerror();
    throw JUBATUS_EXCEPTION(
        converter_exception(
            "cannot load dynamic library: " + path + ": " + error)
        << jubatus::core::common::exception::error_api_func("dlopen")
        << jubatus::core::common::exception::error_file_name(path)
        << jubatus::core::common::exception::error_message(error));
  }
}

dynamic_loader::~dynamic_loader() {
  if (handle_ && dlclose(handle_) != 0) {
    // TODO(kuenishi) failed
  }
}

void* dynamic_loader::load_symbol(const std::string& name) const {
  dlerror();
  void* func = dlsym(handle_, name.c_str());
  char* error = dlerror();
  if (error != NULL) {
    throw JUBATUS_EXCEPTION(converter_exception("cannot dlsym: " + name)
      << jubatus::core::common::exception::error_api_func("dlsym")
      << jubatus::core::common::exception::error_message("dlsym name: " + name)
      << jubatus::core::common::exception::error_message(error));
  }
  return func;
}

void check_null_instance(void* inst) {
  if (!inst) {
    throw JUBATUS_EXCEPTION(converter_exception("failed to load plugin"));
  }
}

}  // namespace fv_converter
}  // namespace core
}  // namespace jubatus
