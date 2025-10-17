/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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

#include <stdint.h>

#include <memory>
#include <vector>

#include <platform/bionic/macros.h>

#include "Config.h"
#include "GuardData.h"
#include "PointerData.h"
#include "RecordData.h"
#include "malloc_debug.h"

class DebugData {
 public:
  DebugData() = default;
  ~DebugData() = default;

  bool Initialize(const char* options);

  static bool Disabled();

  inline void* GetPointer(const Header* header) {
    uintptr_t value = reinterpret_cast<uintptr_t>(header);
    return reinterpret_cast<void*>(value + pointer_offset_);
  }

  Header* GetHeader(const void* pointer) {
    uintptr_t value = reinterpret_cast<uintptr_t>(pointer);
    return reinterpret_cast<Header*>(value - pointer_offset_);
  }

  uint8_t* GetFrontGuard(const Header* header) {
    uintptr_t value = reinterpret_cast<uintptr_t>(header);
    return reinterpret_cast<uint8_t*>(value + front_guard->offset());
  }

  uint8_t* GetRearGuard(const Header* header) {
    uintptr_t value = reinterpret_cast<uintptr_t>(GetPointer(header));
    return reinterpret_cast<uint8_t*>(value + header->size);
  }

  const Config& config() { return config_; }
  size_t pointer_offset() { return pointer_offset_; }
  size_t extra_bytes() { return extra_bytes_; }

  bool TrackPointers() { return config_.options() & TRACK_ALLOCS; }

  bool HeaderEnabled() { return config_.options() & HEADER_OPTIONS; }

  void PrepareFork();
  void PostForkParent();
  void PostForkChild();

  std::unique_ptr<FrontGuardData> front_guard;
  std::unique_ptr<PointerData> pointer;
  std::unique_ptr<RearGuardData> rear_guard;
  std::unique_ptr<RecordData> record;

 private:
  size_t extra_bytes_ = 0;

  size_t pointer_offset_ = 0;

  Config config_;

  BIONIC_DISALLOW_COPY_AND_ASSIGN(DebugData);
};

extern DebugData* g_debug;
