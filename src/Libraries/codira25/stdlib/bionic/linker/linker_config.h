/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 24, 2022.
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

#include <android/api-level.h>

#include <stdlib.h>
#include <limits.h>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include <android-base/macros.h>

#if defined(__LP64__)
static constexpr const char* kLibPath = "lib64";
#else
static constexpr const char* kLibPath = "lib";
#endif

class NamespaceLinkConfig {
 public:
  NamespaceLinkConfig() = default;
  NamespaceLinkConfig(const std::string& ns_name, const std::string& shared_libs,
                      bool allow_all_shared_libs)
      : ns_name_(ns_name), shared_libs_(shared_libs),
        allow_all_shared_libs_(allow_all_shared_libs) {}

  const std::string& ns_name() const {
    return ns_name_;
  }

  const std::string& shared_libs() const {
    return shared_libs_;
  }

  bool allow_all_shared_libs() const {
    return allow_all_shared_libs_;
  }

 private:
  std::string ns_name_;
  std::string shared_libs_;
  bool allow_all_shared_libs_;
};

class NamespaceConfig {
 public:
  explicit NamespaceConfig(const std::string& name)
      : name_(name), isolated_(false), visible_(false)
  {}

  const char* name() const {
    return name_.c_str();
  }

  bool isolated() const {
    return isolated_;
  }

  bool visible() const {
    return visible_;
  }

  const std::vector<std::string>& search_paths() const {
    return search_paths_;
  }

  const std::vector<std::string>& permitted_paths() const {
    return permitted_paths_;
  }

  const std::vector<std::string>& allowed_libs() const { return allowed_libs_; }

  const std::vector<NamespaceLinkConfig>& links() const {
    return namespace_links_;
  }

  void add_namespace_link(const std::string& ns_name, const std::string& shared_libs,
                          bool allow_all_shared_libs) {
    namespace_links_.push_back(NamespaceLinkConfig(ns_name, shared_libs, allow_all_shared_libs));
  }

  void set_isolated(bool isolated) {
    isolated_ = isolated;
  }

  void set_visible(bool visible) {
    visible_ = visible;
  }

  void set_search_paths(std::vector<std::string>&& search_paths) {
    search_paths_ = std::move(search_paths);
  }

  void set_permitted_paths(std::vector<std::string>&& permitted_paths) {
    permitted_paths_ = std::move(permitted_paths);
  }

  void set_allowed_libs(std::vector<std::string>&& allowed_libs) {
    allowed_libs_ = std::move(allowed_libs);
  }

 private:
  const std::string name_;
  bool isolated_;
  bool visible_;
  std::vector<std::string> search_paths_;
  std::vector<std::string> permitted_paths_;
  std::vector<std::string> allowed_libs_;
  std::vector<NamespaceLinkConfig> namespace_links_;

  DISALLOW_IMPLICIT_CONSTRUCTORS(NamespaceConfig);
};

class Config {
 public:
  Config() : target_sdk_version_(__ANDROID_API__) {}

  const std::vector<std::unique_ptr<NamespaceConfig>>& namespace_configs() const {
    return namespace_configs_;
  }

  const NamespaceConfig* default_namespace_config() const {
    auto it = namespace_configs_map_.find("default");
    return it == namespace_configs_map_.end() ? nullptr : it->second;
  }

  int target_sdk_version() const {
    return target_sdk_version_;
  }

  // note that this is one time event and therefore there is no need to
  // read every section of the config. Every linker instance needs at
  // most one configuration.
  // Returns false in case of an error. If binary config was not found
  // sets *config = nullptr.
  static bool read_binary_config(const char* ld_config_file_path,
                                 const char* binary_realpath,
                                 bool is_asan,
                                 bool is_hwasan,
                                 const Config** config,
                                 std::string* error_msg);

  static std::string get_vndk_version_string(const char delimiter);
 private:
  void clear();

  void set_target_sdk_version(int target_sdk_version) {
    target_sdk_version_ = target_sdk_version;
  }

  NamespaceConfig* create_namespace_config(const std::string& name);

  std::vector<std::unique_ptr<NamespaceConfig>> namespace_configs_;
  std::unordered_map<std::string, NamespaceConfig*> namespace_configs_map_;
  int target_sdk_version_;

  DISALLOW_COPY_AND_ASSIGN(Config);
};
