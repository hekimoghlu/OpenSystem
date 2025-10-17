/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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

#include "contexts.h"

struct PrefixNode;
class ContextListNode;

class ContextsSplit : public Contexts {
 public:
  virtual ~ContextsSplit() override {
  }

  virtual bool Initialize(bool writable, const char* filename, bool* fsetxattr_failed,
                          bool) override;
  virtual prop_area* GetPropAreaForName(const char* name) override;
  virtual prop_area* GetSerialPropArea() override {
    return serial_prop_area_;
  }
  virtual void ForEach(void (*propfn)(const prop_info* pi, void* cookie), void* cookie) override;
  virtual void ResetAccess() override;
  virtual void FreeAndUnmap() override;

  PrefixNode* GetPrefixNodeForName(const char* name);

 protected:
  bool MapSerialPropertyArea(bool access_rw, bool* fsetxattr_failed);
  bool InitializePropertiesFromFile(const char* filename);
  bool InitializeProperties();

  PrefixNode* prefixes_ = nullptr;
  ContextListNode* contexts_ = nullptr;
  prop_area* serial_prop_area_ = nullptr;
  const char* filename_ = nullptr;
};
