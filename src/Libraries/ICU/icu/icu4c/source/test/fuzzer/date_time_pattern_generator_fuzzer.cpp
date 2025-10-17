/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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

#include <cstring>

#include "fuzzer_utils.h"
#include "unicode/dtptngen.h"
#include "unicode/localpointer.h"
#include "unicode/locid.h"

IcuEnvironment* env = new IcuEnvironment();

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  UErrorCode status = U_ZERO_ERROR;

  uint16_t rnd16;

  if (size < 2 + sizeof(rnd16))
    return 0;

  std::memcpy(&rnd16, data, sizeof(rnd16));
  size -= sizeof(rnd16);
  data += sizeof(rnd16);
  const icu::Locale& locale = GetRandomLocale(rnd16);

  std::unique_ptr<char16_t[]> fuzzbuff(new char16_t[size/2]);
  std::memcpy(fuzzbuff.get(), data, (size/2)*2);
  icu::UnicodeString fuzzstr(false, fuzzbuff.get(), size/2);

  icu::LocalPointer<icu::DateTimePatternGenerator > gen(
      icu::DateTimePatternGenerator::createInstance(locale, status), status);
  if (U_SUCCESS(status)) {

      status = U_ZERO_ERROR;
      gen->getSkeleton(fuzzstr, status);

      status = U_ZERO_ERROR;
      gen->getBaseSkeleton(fuzzstr, status);

      status = U_ZERO_ERROR;
      gen->getBaseSkeleton(fuzzstr, status);

      status = U_ZERO_ERROR;
      gen->getPatternForSkeleton(fuzzstr);

      status = U_ZERO_ERROR;
      gen->getBestPattern(fuzzstr, status);

      status = U_ZERO_ERROR;
      icu::DateTimePatternGenerator::staticGetSkeleton(fuzzstr, status);

      status = U_ZERO_ERROR;
      icu::DateTimePatternGenerator::staticGetBaseSkeleton (fuzzstr, status);
  }

  status = U_ZERO_ERROR;
  std::string str(reinterpret_cast<const char*>(data), size);
  gen.adoptInstead(icu::DateTimePatternGenerator::createInstance(icu::Locale(str.c_str()), status));
  return 0;
}
