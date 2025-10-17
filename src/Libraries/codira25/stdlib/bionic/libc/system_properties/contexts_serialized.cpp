/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#include "system_properties/contexts_serialized.h"

#include <fcntl.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <new>

#include <async_safe/log.h>
#include <private/android_filesystem_config.h>

#include "system_properties/system_properties.h"

bool ContextsSerialized::InitializeContextNodes() {
  auto num_context_nodes = property_info_area_file_->num_contexts();
  auto context_nodes_mmap_size = sizeof(ContextNode) * num_context_nodes;
  // We want to avoid malloc in system properties, so we take an anonymous map instead (b/31659220).
  void* const map_result = mmap(nullptr, context_nodes_mmap_size, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (map_result == MAP_FAILED) {
    return false;
  }

  prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, map_result, context_nodes_mmap_size,
        "System property context nodes");

  context_nodes_ = reinterpret_cast<ContextNode*>(map_result);
  num_context_nodes_ = num_context_nodes;
  context_nodes_mmap_size_ = context_nodes_mmap_size;

  for (size_t i = 0; i < num_context_nodes; ++i) {
    new (&context_nodes_[i]) ContextNode(property_info_area_file_->context(i), dirname_);
  }

  return true;
}

bool ContextsSerialized::MapSerialPropertyArea(bool access_rw, bool* fsetxattr_failed) {
  if (access_rw) {
    serial_prop_area_ = prop_area::map_prop_area_rw(
        serial_filename_.c_str(), "u:object_r:properties_serial:s0", fsetxattr_failed);
  } else {
    serial_prop_area_ = prop_area::map_prop_area(serial_filename_.c_str());
  }
  return serial_prop_area_;
}

// Note: load_default_path is only used for testing, as it will cause properties to be loaded from
// one file (specified by PropertyInfoAreaFile.LoadDefaultPath), but be written to "filename".
bool ContextsSerialized::InitializeProperties(bool load_default_path) {
  if (load_default_path && !property_info_area_file_.LoadDefaultPath()) {
    return false;
  } else if (!load_default_path && !property_info_area_file_.LoadPath(tree_filename_.c_str())) {
    return false;
  }

  if (!InitializeContextNodes()) {
    FreeAndUnmap();
    return false;
  }

  return true;
}

// Note: load_default_path is only used for testing, as it will cause properties to be loaded from
// one file (specified by PropertyInfoAreaFile.LoadDefaultPath), but be written to "filename".
bool ContextsSerialized::Initialize(bool writable, const char* dirname, bool* fsetxattr_failed,
                                    bool load_default_path) {
  dirname_ = dirname;
  tree_filename_ = PropertiesFilename(dirname, "property_info");
  serial_filename_ = PropertiesFilename(dirname, "properties_serial");

  if (!InitializeProperties(load_default_path)) {
    return false;
  }

  if (writable) {
    mkdir(dirname_, S_IRWXU | S_IXGRP | S_IXOTH);
    bool open_failed = false;
    if (fsetxattr_failed) {
      *fsetxattr_failed = false;
    }

    for (size_t i = 0; i < num_context_nodes_; ++i) {
      if (!context_nodes_[i].Open(true, fsetxattr_failed)) {
        open_failed = true;
      }
    }
    if (open_failed || !MapSerialPropertyArea(true, fsetxattr_failed)) {
      FreeAndUnmap();
      return false;
    }
  } else {
    if (!MapSerialPropertyArea(false, nullptr)) {
      FreeAndUnmap();
      return false;
    }
  }
  return true;
}

prop_area* ContextsSerialized::GetPropAreaForName(const char* name) {
  uint32_t index;
  property_info_area_file_->GetPropertyInfoIndexes(name, &index, nullptr);
  if (index == ~0u || index >= num_context_nodes_) {
    async_safe_format_log(ANDROID_LOG_ERROR, "libc", "Could not find context for property \"%s\"",
                          name);
    return nullptr;
  }
  auto* context_node = &context_nodes_[index];
  if (!context_node->pa()) {
    // We explicitly do not check no_access_ in this case because unlike the
    // case of foreach(), we want to generate an selinux audit for each
    // non-permitted property access in this function.
    context_node->Open(false, nullptr);
  }
  return context_node->pa();
}

void ContextsSerialized::ForEach(void (*propfn)(const prop_info* pi, void* cookie), void* cookie) {
  for (size_t i = 0; i < num_context_nodes_; ++i) {
    if (context_nodes_[i].CheckAccessAndOpen()) {
      context_nodes_[i].pa()->foreach (propfn, cookie);
    }
  }
}

void ContextsSerialized::ResetAccess() {
  for (size_t i = 0; i < num_context_nodes_; ++i) {
    context_nodes_[i].ResetAccess();
  }
}

void ContextsSerialized::FreeAndUnmap() {
  property_info_area_file_.Reset();
  if (context_nodes_ != nullptr) {
    for (size_t i = 0; i < num_context_nodes_; ++i) {
      context_nodes_[i].Unmap();
    }
    munmap(context_nodes_, context_nodes_mmap_size_);
    context_nodes_ = nullptr;
  }
  prop_area::unmap_prop_area(&serial_prop_area_);
  serial_prop_area_ = nullptr;
}
