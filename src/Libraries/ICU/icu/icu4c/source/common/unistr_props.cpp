/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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
/*
*******************************************************************************
*
*   Copyright (C) 1999-2011, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
*******************************************************************************
*   file name:  unistr_props.cpp
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:2
*
*   created on: 2004aug25
*   created by: Markus W. Scherer
*
*   Character property dependent functions moved here from unistr.cpp
*/

#include "unicode/utypes.h"
#include "unicode/uchar.h"
#include "unicode/unistr.h"
#include "unicode/utf16.h"

U_NAMESPACE_BEGIN

UnicodeString& 
UnicodeString::trim()
{
  if(isBogus()) {
    return *this;
  }

  char16_t *array = getArrayStart();
  UChar32 c;
  int32_t oldLength = this->length();
  int32_t i = oldLength, length;

  // first cut off trailing white space
  for(;;) {
    length = i;
    if(i <= 0) {
      break;
    }
    U16_PREV(array, 0, i, c);
    if(!(c == 0x20 || u_isWhitespace(c))) {
      break;
    }
  }
  if(length < oldLength) {
    setLength(length);
  }

  // find leading white space
  int32_t start;
  i = 0;
  for(;;) {
    start = i;
    if(i >= length) {
      break;
    }
    U16_NEXT(array, i, length, c);
    if(!(c == 0x20 || u_isWhitespace(c))) {
      break;
    }
  }

  // move string forward over leading white space
  if(start > 0) {
    doReplace(0, start, nullptr, 0, 0);
  }

  return *this;
}

U_NAMESPACE_END
