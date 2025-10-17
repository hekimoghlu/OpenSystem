/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#pragma once

#include "linker_common_types.h"

#include <string>
#include <vector>
#include <unordered_set>

std::vector<std::string> fix_lib_paths(std::vector<std::string> paths);

struct android_namespace_t;

struct android_namespace_link_t {
 public:
  android_namespace_link_t(android_namespace_t* linked_namespace,
                           std::unordered_set<std::string> shared_lib_sonames,
                           bool allow_all_shared_libs)
      : linked_namespace_(linked_namespace),
        shared_lib_sonames_(std::move(shared_lib_sonames)),
        allow_all_shared_libs_(allow_all_shared_libs) {}

  android_namespace_t* linked_namespace() const {
    return linked_namespace_;
  }

  const std::unordered_set<std::string>& shared_lib_sonames() const {
    return shared_lib_sonames_;
  }

  bool is_accessible(const char* soname) const {
    return allow_all_shared_libs_ || shared_lib_sonames_.find(soname) != shared_lib_sonames_.end();
  }

  bool allow_all_shared_libs() const {
    return allow_all_shared_libs_;
  }

 private:
  android_namespace_t* const linked_namespace_;
  const std::unordered_set<std::string> shared_lib_sonames_;
  bool allow_all_shared_libs_;
};

struct android_namespace_t {
 public:
  android_namespace_t() :
    is_isolated_(false),
    is_exempt_list_enabled_(false),
    is_also_used_as_anonymous_(false) {}

  const char* get_name() const { return name_.c_str(); }
  void set_name(const char* name) { name_ = name; }

  bool is_isolated() const { return is_isolated_; }
  void set_isolated(bool isolated) { is_isolated_ = isolated; }

  bool is_exempt_list_enabled() const { return is_exempt_list_enabled_; }
  void set_exempt_list_enabled(bool enabled) { is_exempt_list_enabled_ = enabled; }

  bool is_also_used_as_anonymous() const { return is_also_used_as_anonymous_; }
  void set_also_used_as_anonymous(bool yes) { is_also_used_as_anonymous_ = yes; }

  const std::vector<std::string>& get_ld_library_paths() const {
    return ld_library_paths_;
  }
  void set_ld_library_paths(std::vector<std::string>&& library_paths) {
    ld_library_paths_ = std::move(library_paths);
  }

  const std::vector<std::string>& get_default_library_paths() const {
    return default_library_paths_;
  }
  void set_default_library_paths(std::vector<std::string>&& library_paths) {
    default_library_paths_ = fix_lib_paths(std::move(library_paths));
  }
  void set_default_library_paths(const std::vector<std::string>& library_paths) {
    default_library_paths_ = fix_lib_paths(library_paths);
  }

  const std::vector<std::string>& get_permitted_paths() const {
    return permitted_paths_;
  }
  void set_permitted_paths(std::vector<std::string>&& permitted_paths) {
    permitted_paths_ = std::move(permitted_paths);
  }
  void set_permitted_paths(const std::vector<std::string>& permitted_paths) {
    permitted_paths_ = permitted_paths;
  }

  const std::vector<std::string>& get_allowed_libs() const { return allowed_libs_; }
  void set_allowed_libs(std::vector<std::string>&& allowed_libs) {
    allowed_libs_ = std::move(allowed_libs);
  }
  void set_allowed_libs(const std::vector<std::string>& allowed_libs) {
    allowed_libs_ = allowed_libs;
  }

  const std::vector<android_namespace_link_t>& linked_namespaces() const {
    return linked_namespaces_;
  }
  void add_linked_namespace(android_namespace_t* linked_namespace,
                            std::unordered_set<std::string> shared_lib_sonames,
                            bool allow_all_shared_libs) {
    linked_namespaces_.emplace_back(linked_namespace, std::move(shared_lib_sonames),
                                    allow_all_shared_libs);
  }

  void add_soinfo(soinfo* si) {
    soinfo_list_.push_back(si);
  }

  void add_soinfos(const soinfo_list_t& soinfos) {
    for (auto si : soinfos) {
      add_soinfo(si);
    }
  }

  void remove_soinfo(soinfo* si) {
    soinfo_list_.remove_if([&](soinfo* candidate) {
      return si == candidate;
    });
  }

  const soinfo_list_t& soinfo_list() const { return soinfo_list_; }

  // For isolated namespaces - checks if the file is on the search path;
  // always returns true for not isolated namespace.
  bool is_accessible(const std::string& path);

  // Returns true if si is accessible from this namespace. A soinfo
  // is considered accessible when it belongs to this namespace
  // or one of it's parent soinfos belongs to this namespace.
  bool is_accessible(soinfo* si);

  soinfo_list_t get_global_group();
  soinfo_list_t get_shared_group();

 private:
  std::string name_;
  bool is_isolated_;
  bool is_exempt_list_enabled_;
  bool is_also_used_as_anonymous_;
  std::vector<std::string> ld_library_paths_;
  std::vector<std::string> default_library_paths_;
  std::vector<std::string> permitted_paths_;
  std::vector<std::string> allowed_libs_;
  // Loader looks into linked namespace if it was not able
  // to find a library in this namespace. Note that library
  // lookup in linked namespaces are limited by the list of
  // shared sonames.
  std::vector<android_namespace_link_t> linked_namespaces_;
  soinfo_list_t soinfo_list_;

  DISALLOW_COPY_AND_ASSIGN(android_namespace_t);
};
