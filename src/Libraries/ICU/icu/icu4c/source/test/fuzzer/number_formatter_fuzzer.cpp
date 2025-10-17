/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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
#include "unicode/localpointer.h"
#include "unicode/numberformatter.h"

IcuEnvironment* env = new IcuEnvironment();

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  UErrorCode status = U_ZERO_ERROR;

  int16_t rnd;
  int64_t value;
  double doubleValue;
  if (size < sizeof(rnd) + sizeof(value) + sizeof(doubleValue)) return 0;
  icu::StringPiece fuzzData(reinterpret_cast<const char *>(data), size);

  std::memcpy(&rnd, fuzzData.data(), sizeof(rnd));
  icu::Locale locale = GetRandomLocale(rnd);
  fuzzData.remove_prefix(sizeof(rnd));

  std::memcpy(&value, fuzzData.data(), sizeof(value));
  fuzzData.remove_prefix(sizeof(value));

  std::memcpy(&doubleValue, fuzzData.data(), sizeof(doubleValue));
  fuzzData.remove_prefix(sizeof(doubleValue));

  size_t len = fuzzData.size() / sizeof(char16_t);
  icu::UnicodeString fuzzstr(false, reinterpret_cast<const char16_t*>(fuzzData.data()), len);

  icu::number::UnlocalizedNumberFormatter unf = icu::number::NumberFormatter::forSkeleton(
      fuzzstr, status);

  icu::number::LocalizedNumberFormatter nf = unf.locale(locale);

  status = U_ZERO_ERROR;
  nf.formatInt(value, status);

  status = U_ZERO_ERROR;
  nf.formatDouble(doubleValue, status);
  return 0;
}
