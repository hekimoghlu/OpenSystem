/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 24, 2025.
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

// Fuzzer for ICU Locales.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "unicode/locid.h"

namespace {

void ConsumeNBytes(const uint8_t** data, size_t* size, size_t N) {
  *data += N;
  *size -= N;
}

uint8_t ConsumeUint8(const uint8_t** data, size_t* size) {
  uint8_t tmp = 0;
  if (*size >= 1) {
    tmp = (*data)[0];
    ConsumeNBytes(data, size, 1);
  }
  return tmp;
}

std::string ConsumeSubstring(const uint8_t** data, size_t* size) {
  const size_t request_size = ConsumeUint8(data, size);
  const char* substring_start = reinterpret_cast<const char*>(*data);
  const size_t substring_size = std::min(*size, request_size);
  ConsumeNBytes(data, size, substring_size);
  return std::string(substring_start, substring_size);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  const std::string language = ConsumeSubstring(&data, &size);
  const std::string country = ConsumeSubstring(&data, &size);
  const std::string variant = ConsumeSubstring(&data, &size);
  const std::string kv_pairs = ConsumeSubstring(&data, &size);
  icu::Locale locale(language.c_str(), country.c_str(), variant.c_str(),
                     kv_pairs.c_str());
  return EXIT_SUCCESS;
}
