/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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

#include <property_info_parser/property_info_parser.h>

#include "context_node.h"
#include "contexts.h"
#include "properties_filename.h"

class ContextsSerialized : public Contexts {
 public:
  virtual ~ContextsSerialized() override {
  }

  virtual bool Initialize(bool writable, const char* dirname, bool* fsetxattr_failed,
                          bool load_default_path) override;
  virtual prop_area* GetPropAreaForName(const char* name) override;
  virtual prop_area* GetSerialPropArea() override {
    return serial_prop_area_;
  }
  virtual void ForEach(void (*propfn)(const prop_info* pi, void* cookie), void* cookie) override;
  virtual void ResetAccess() override;
  virtual void FreeAndUnmap() override;

 private:
  bool InitializeContextNodes();
  bool InitializeProperties(bool load_default_path);
  bool MapSerialPropertyArea(bool access_rw, bool* fsetxattr_failed);

  const char* dirname_;
  PropertiesFilename tree_filename_;
  PropertiesFilename serial_filename_;
  android::properties::PropertyInfoAreaFile property_info_area_file_;
  ContextNode* context_nodes_ = nullptr;
  size_t num_context_nodes_ = 0;
  size_t context_nodes_mmap_size_ = 0;
  prop_area* serial_prop_area_ = nullptr;
};
