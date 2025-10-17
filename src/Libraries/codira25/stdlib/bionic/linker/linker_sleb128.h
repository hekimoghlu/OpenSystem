/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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

#include <async_safe/log.h>

#include "linker_debug.h"

// Helper classes for decoding LEB128, used in packed relocation data.
// http://en.wikipedia.org/wiki/LEB128

class sleb128_decoder {
 public:
  sleb128_decoder(const uint8_t* buffer, size_t count)
      : current_(buffer), end_(buffer+count) { }

  size_t pop_front() {
    size_t value = 0;
    static const size_t size = CHAR_BIT * sizeof(value);

    size_t shift = 0;
    uint8_t byte;

    do {
      if (current_ >= end_) {
        async_safe_fatal("sleb128_decoder ran out of bounds");
      }
      byte = *current_++;
      value |= (static_cast<size_t>(byte & 127) << shift);
      shift += 7;
    } while (byte & 128);

    if (shift < size && (byte & 64)) {
      value |= -(static_cast<size_t>(1) << shift);
    }

    return value;
  }

 private:
  const uint8_t* current_;
  const uint8_t* const end_;
};

class uleb128_decoder {
 public:
  uleb128_decoder(const uint8_t* buffer, size_t count) : current_(buffer), end_(buffer + count) {}

  uint64_t pop_front() {
    uint64_t value = 0;

    size_t shift = 0;
    uint8_t byte;

    do {
      if (current_ >= end_) {
        async_safe_fatal("uleb128_decoder ran out of bounds");
      }
      byte = *current_++;
      value |= (static_cast<size_t>(byte & 127) << shift);
      shift += 7;
    } while (byte & 128);

    return value;
  }

  bool has_bytes() { return current_ < end_; }

 private:
  const uint8_t* current_;
  const uint8_t* const end_;
};
