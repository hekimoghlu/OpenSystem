/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/**
 * Copyright (c) 1999-2012, International Business Machines Corporation and
 * others. All Rights Reserved.
 *
 * Test for source/i18n/collunsafe.h
 */

#include <stdio.h>
#include "unicode/ucol.h"
#include "unicode/uniset.h"
#include "unicode/coll.h"
#include "collation.h"

#include "collunsafe.h"

using icu::Collator;
using icu::Locale;
using icu::UnicodeSet;

int main(int argc, const char *argv[]) {
  puts("verify");
  UErrorCode errorCode = U_ZERO_ERROR;
#if defined (COLLUNSAFE_PATTERN)
  puts("verify pattern");
  const UnicodeString unsafeBackwardPattern(false, collunsafe_pattern, collunsafe_len);
  fprintf(stderr, "\n -- pat '%c%c%c%c%c'\n",
          collunsafe_pattern[0],
          collunsafe_pattern[1],
          collunsafe_pattern[2],
          collunsafe_pattern[3],
          collunsafe_pattern[4]);
  if(U_SUCCESS(errorCode)) {
    UnicodeSet us(unsafeBackwardPattern, errorCode);
    fprintf(stderr, "\n%s:%d: err creating set %s\n", __FILE__, __LINE__, u_errorName(errorCode));
  }
#endif

#if defined (COLLUNSAFE_RANGE)
  {
    puts("verify range");
    UnicodeSet u;
    for(int32_t i=0;i<unsafe_rangeCount*2;i+=2) {
      u.add(unsafe_ranges[i+0],unsafe_ranges[i+1]);
    }
    printf("Finished with %d ranges\n", u.getRangeCount());
  }
#endif

#if defined (COLLUNSAFE_SERIALIZE)
  {
    puts("verify serialize");
    UnicodeSet u(unsafe_serializedData, unsafe_serializedCount, UnicodeSet::kSerialized, errorCode);
    fprintf(stderr, "\n%s:%d: err creating set %s\n", __FILE__, __LINE__, u_errorName(errorCode));
    printf("Finished deserialize with %d ranges\n", u.getRangeCount());
  }
#endif
// if(tailoring.unsafeBackwardSet == nullptr) {
  //   errorCode = U_MEMORY_ALLOCATION_ERROR;
  //   fprintf(stderr, "\n%s:%d: err %s\n", __FILE__, __LINE__, u_errorName(errorCode));
  // }
  puts("verify col UCA");
  if(U_SUCCESS(errorCode)) {
    Collator *col = Collator::createInstance(Locale::getEnglish(), errorCode);
    fprintf(stderr, "\n%s:%d: err %s creating collator\n", __FILE__, __LINE__, u_errorName(errorCode));
  }
  
  if(U_FAILURE(errorCode)) {
    return 1;
  } else {
    return 0;
  }
}
