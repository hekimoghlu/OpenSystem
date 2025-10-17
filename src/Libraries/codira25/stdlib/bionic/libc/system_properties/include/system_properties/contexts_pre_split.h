/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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
#include "prop_area.h"
#include "prop_info.h"

class ContextsPreSplit : public Contexts {
 public:
  virtual ~ContextsPreSplit() override {
  }

  // We'll never initialize this legacy option as writable, so don't even check the arg.
  virtual bool Initialize(bool, const char* filename, bool*, bool) override {
    pre_split_prop_area_ = prop_area::map_prop_area(filename);
    return pre_split_prop_area_ != nullptr;
  }

  virtual prop_area* GetPropAreaForName(const char*) override {
    return pre_split_prop_area_;
  }

  virtual prop_area* GetSerialPropArea() override {
    return pre_split_prop_area_;
  }

  virtual void ForEach(void (*propfn)(const prop_info* pi, void* cookie), void* cookie) override {
    pre_split_prop_area_->foreach (propfn, cookie);
  }

  // This is a no-op for pre-split properties as there is only one property file and it is
  // accessible by all domains
  virtual void ResetAccess() override {
  }

  virtual void FreeAndUnmap() override {
    prop_area::unmap_prop_area(&pre_split_prop_area_);
  }

 private:
  prop_area* pre_split_prop_area_ = nullptr;
};
