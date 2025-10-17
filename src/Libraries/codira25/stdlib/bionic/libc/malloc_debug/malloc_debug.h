/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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

#include <private/bionic_malloc_dispatch.h>

// Allocations that require a header include a variable length header.
// This is the order that data structures will be found. If an optional
// part of the header does not exist, the other parts of the header
// will still be in this order.
//   Header          (Required)
//   uint8_t data    (Optional: Front guard, will be a multiple of MINIMUM_ALIGNMENT_BYTES)
//   allocation data
//   uint8_t data    (Optional: End guard)
//
// In the initialization function, offsets into the header will be set
// for each different header location. The offsets are always from the
// beginning of the Header section.
struct Header {
  uint32_t tag;
  void* orig_pointer;
  size_t size;
  size_t usable_size;
} __attribute__((packed));

struct BacktraceHeader {
  size_t num_frames;
  uintptr_t frames[0];
} __attribute__((packed));

constexpr uint32_t DEBUG_TAG = 0x1ee7d00d;
constexpr uint32_t DEBUG_FREE_TAG = 0x1cc7dccd;
constexpr char LOG_DIVIDER[] = "*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***";
constexpr size_t FREE_TRACK_MEM_BUFFER_SIZE = 4096;

extern const MallocDispatch* g_dispatch;

void BacktraceAndLog();
