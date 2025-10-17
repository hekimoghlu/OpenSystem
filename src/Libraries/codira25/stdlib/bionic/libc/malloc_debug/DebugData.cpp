/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
#include <stdint.h>

#include "Config.h"
#include "DebugData.h"
#include "GuardData.h"
#include "LogAllocatorStats.h"
#include "PointerData.h"
#include "debug_disable.h"
#include "malloc_debug.h"

bool DebugData::Initialize(const char* options) {
  if (!config_.Init(options)) {
    return false;
  }

  // Check to see if the options that require a header are enabled.
  if (config_.options() & HEADER_OPTIONS) {
    // Initialize all of the static header offsets.
    pointer_offset_ = __BIONIC_ALIGN(sizeof(Header), MINIMUM_ALIGNMENT_BYTES);

    if (config_.options() & FRONT_GUARD) {
      front_guard.reset(new FrontGuardData(this, config_, &pointer_offset_));
    }

    extra_bytes_ = pointer_offset_;

    // Initialize all of the non-header data.
    if (config_.options() & REAR_GUARD) {
      rear_guard.reset(new RearGuardData(this, config_));
      extra_bytes_ += config_.rear_guard_bytes();
    }
  }

  if (TrackPointers()) {
    pointer.reset(new PointerData(this));
    if (!pointer->Initialize(config_)) {
      return false;
    }
  }

  if (config_.options() & RECORD_ALLOCS) {
    record.reset(new RecordData());
    if (!record->Initialize(config_)) {
      return false;
    }
  }

  if (config_.options() & EXPAND_ALLOC) {
    extra_bytes_ += config_.expand_alloc_bytes();
  }

  if (config_.options() & LOG_ALLOCATOR_STATS_ON_SIGNAL) {
    if (!LogAllocatorStats::Initialize(config_)) {
      return false;
    }
  }

  return true;
}

void DebugData::PrepareFork() {
  if (pointer != nullptr) {
    pointer->PrepareFork();
  }
}

void DebugData::PostForkParent() {
  if (pointer != nullptr) {
    pointer->PostForkParent();
  }
}

void DebugData::PostForkChild() {
  if (pointer != nullptr) {
    pointer->PostForkChild();
  }
}
