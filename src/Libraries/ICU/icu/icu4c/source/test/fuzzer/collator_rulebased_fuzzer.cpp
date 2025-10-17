/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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
#include "unicode/coll.h"
#include "unicode/localpointer.h"
#include "unicode/locid.h"
#include "unicode/sortkey.h"
#include "unicode/tblcoll.h"

IcuEnvironment* env = new IcuEnvironment();

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  UErrorCode status = U_ZERO_ERROR;
  if (size > 2000) {
      // Limit the effective test data to only 2000 bytes to avoid meaningless
      // timeout.
      size = 2000;
  }

#if APPLE_ICU_CHANGES
// rdar://70810661 (ability to compile ICU with asan and libfuzzer)
  icu::UnicodeString fuzzstr(false, reinterpret_cast<const UChar*>(data), size / 2);
#else
  size_t unistr_size = size/2;
  std::unique_ptr<char16_t[]> fuzzbuff(new char16_t[unistr_size]);
  std::memcpy(fuzzbuff.get(), data, unistr_size * 2);
  icu::UnicodeString fuzzstr(false, fuzzbuff.get(), unistr_size);
#endif  // APPLE_ICU_CHANGES

  icu::LocalPointer<icu::RuleBasedCollator> col1(
      new icu::RuleBasedCollator(fuzzstr, status));

  if (U_SUCCESS(status)) {
      col1->getVariableTop(status);
      status = U_ZERO_ERROR;
      icu::CollationKey key;
      col1->getCollationKey(fuzzstr, key, status);
      status = U_ZERO_ERROR;
      icu::LocalPointer<icu::UnicodeSet> tailoredSet(col1->getTailoredSet(status));
      status = U_ZERO_ERROR;
      col1->getLocale(ULOC_ACTUAL_LOCALE, status);
      status = U_ZERO_ERROR;
      col1->getLocale(ULOC_VALID_LOCALE, status);
      col1->getMaxVariable();
      col1->getStrength();
      col1->getSortKey(fuzzstr, nullptr, 0);
  }
  return 0;
}
