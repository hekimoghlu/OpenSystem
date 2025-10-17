/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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

// Â© 2019 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

// Fuzzer for ucasemap.

#include <cstring>
#include <functional>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include "fuzzer_utils.h"
#include "unicode/ucasemap.h"

IcuEnvironment* env = new IcuEnvironment();

template<typename T>
using deleted_unique_ptr = std::unique_ptr<T,std::function<void(T*)>>;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  UErrorCode status = U_ZERO_ERROR;
  uint8_t rnd8 = 0;
  uint16_t rnd16 = 0;
  uint32_t rnd32 = 0;

  if (size < 7) {
    return 0;
  }
  // Extract one, two, and four bytes from fuzzer data for random selection
  // purposes.
  rnd8 = *data;
  data++;
  rnd16 = *(reinterpret_cast<const uint16_t *>(data));
  data = data + 2;
  rnd32 = *(reinterpret_cast<const uint32_t *>(data));
  data = data + 4;
  size = size - 7;

  std::unique_ptr<char[]> fuzzbuff(new char[size]);
  std::memcpy(fuzzbuff.get(), data, size);

  const icu::Locale& locale = GetRandomLocale(rnd16);
  uint32_t open_flags = rnd32;

  deleted_unique_ptr<UCaseMap> csm(
      ucasemap_open(locale.getName(), open_flags, &status),
      [](UCaseMap* map) { ucasemap_close(map); });

  if (U_FAILURE(status)) {
    return 0;
  }

  int32_t dst_size = size * 2;
  std::unique_ptr<char[]> dst(new char[dst_size]);
  const auto* src = reinterpret_cast<const char*>(fuzzbuff.get());

  switch (rnd8 % 4) {
    case 0: ucasemap_utf8ToLower(csm.get(), dst.get(), dst_size, src, size,
                &status);
            break;
    case 1: ucasemap_utf8ToUpper(csm.get(), dst.get(), dst_size, src, size,
                &status);
            break;
    case 2: ucasemap_utf8ToTitle(csm.get(), dst.get(), dst_size, src, size,
                &status);
            break;
    case 3: ucasemap_utf8FoldCase(csm.get(), dst.get(), dst_size, src, size,
                &status);
            break;
  }

  return 0;
}
