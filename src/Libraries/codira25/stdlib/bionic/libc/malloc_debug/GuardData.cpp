/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
#include <string.h>

#include <vector>

#include "Config.h"
#include "DebugData.h"
#include "GuardData.h"
#include "backtrace.h"
#include "debug_disable.h"
#include "debug_log.h"
#include "malloc_debug.h"

GuardData::GuardData(DebugData* debug_data, int init_value, size_t num_bytes)
    : OptionData(debug_data) {
  // Create a buffer for fast comparisons of the front guard.
  cmp_mem_.resize(num_bytes);
  memset(cmp_mem_.data(), init_value, cmp_mem_.size());
}

void GuardData::LogFailure(const Header* header, const void* pointer, const void* data) {
  error_log(LOG_DIVIDER);
  error_log("+++ ALLOCATION %p SIZE %zu HAS A CORRUPTED %s GUARD", pointer, header->size,
            GetTypeName());

  // Log all of the failing bytes.
  const uint8_t* expected = cmp_mem_.data();
  int pointer_idx = reinterpret_cast<uintptr_t>(data) - reinterpret_cast<uintptr_t>(pointer);
  const uint8_t* real = reinterpret_cast<const uint8_t*>(data);
  for (size_t i = 0; i < cmp_mem_.size(); i++, pointer_idx++) {
    if (real[i] != expected[i]) {
      error_log("  allocation[%d] = 0x%02x (expected 0x%02x)", pointer_idx, real[i], expected[i]);
    }
  }

  error_log("Backtrace at time of failure:");
  BacktraceAndLog();
  error_log(LOG_DIVIDER);
  if (g_debug->config().options() & ABORT_ON_ERROR) {
    abort();
  }
}

FrontGuardData::FrontGuardData(DebugData* debug_data, const Config& config, size_t* offset)
    : GuardData(debug_data, config.front_guard_value(), config.front_guard_bytes()) {
  // Create a buffer for fast comparisons of the front guard.
  cmp_mem_.resize(config.front_guard_bytes());
  memset(cmp_mem_.data(), config.front_guard_value(), cmp_mem_.size());
  // Assumes that front_bytes is a multiple of MINIMUM_ALIGNMENT_BYTES.
  offset_ = *offset;
  *offset += config.front_guard_bytes();
}

bool FrontGuardData::Valid(const Header* header) {
  return GuardData::Valid(debug_->GetFrontGuard(header));
}

void FrontGuardData::LogFailure(const Header* header) {
  GuardData::LogFailure(header, debug_->GetPointer(header), debug_->GetFrontGuard(header));
}

RearGuardData::RearGuardData(DebugData* debug_data, const Config& config)
    : GuardData(debug_data, config.rear_guard_value(), config.rear_guard_bytes()) {}

bool RearGuardData::Valid(const Header* header) {
  return GuardData::Valid(debug_->GetRearGuard(header));
}

void RearGuardData::LogFailure(const Header* header) {
  GuardData::LogFailure(header, debug_->GetPointer(header), debug_->GetRearGuard(header));
}
