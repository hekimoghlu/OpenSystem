/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "fuzzer_utils.h"
#include "unicode/unistr.h"
#include "unicode/ucnv.h"

IcuEnvironment* env = new IcuEnvironment();

template <typename T>
using deleted_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  UErrorCode status = U_ZERO_ERROR;
  uint16_t rnd = 0;

  if (size < 2) {
    return 0;
  }

  rnd = *(reinterpret_cast<const uint16_t *>(data));
  data = data + 2;
  size = size - 2;

#if APPLE_ICU_CHANGES
// rdar://70810661 (ability to compile ICU with asan and libfuzzer)
  icu::UnicodeString fuzzstr(false, reinterpret_cast<const UChar*>(data), size / 2);
#else
  size_t unistr_size = size/2;
  std::unique_ptr<char16_t[]> fuzzbuff(new char16_t[unistr_size]);
  std::memcpy(fuzzbuff.get(), data, unistr_size * 2);
  
  icu::UnicodeString fuzzstr(false, fuzzbuff.get(), unistr_size);
#endif  // APPLE_ICU_CHANGES

  const char* converter_name =
      ucnv_getAvailableName(rnd % ucnv_countAvailable());
  deleted_unique_ptr<UConverter> converter(ucnv_open(converter_name, &status),
                                           &ucnv_close);
  if (U_FAILURE(status)) {
    return 0;
  }

  static const size_t dest_buffer_size = 1024 * 1204;
  static const std::unique_ptr<char[]> dest_buffer(new char[dest_buffer_size]);

  fuzzstr.extract(dest_buffer.get(), dest_buffer_size, converter.get(), status);

  return 0;
}
