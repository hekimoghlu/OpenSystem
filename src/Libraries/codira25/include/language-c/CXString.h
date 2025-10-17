/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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
#ifndef LANGUAGE_CORE_C_CXSTRING_H
#define LANGUAGE_CORE_C_CXSTRING_H

#include "language/Core-c/ExternC.h"
#include "language/Core-c/Platform.h"

LANGUAGE_CORE_C_EXTERN_C_BEGIN

/**
 * \defgroup CINDEX_STRING String manipulation routines
 * \ingroup CINDEX
 *
 * @{
 */

/**
 * A character string.
 *
 * The \c CXString type is used to return strings from the interface when
 * the ownership of that string might differ from one call to the next.
 * Use \c clang_getCString() to retrieve the string data and, once finished
 * with the string data, call \c clang_disposeString() to free the string.
 */
typedef struct {
  const void *data;
  unsigned private_flags;
} CXString;

typedef struct {
  CXString *Strings;
  unsigned Count;
} CXStringSet;

/**
 * Retrieve the character data associated with the given string.
 *
 * The returned data is a reference and not owned by the user. This data
 * is only valid while the `CXString` is valid. This function is similar
 * to `std::string::c_str()`.
 */
CINDEX_LINKAGE const char *clang_getCString(CXString string);

/**
 * Free the given string.
 */
CINDEX_LINKAGE void clang_disposeString(CXString string);

/**
 * Free the given string set.
 */
CINDEX_LINKAGE void clang_disposeStringSet(CXStringSet *set);

/**
 * @}
 */

LANGUAGE_CORE_C_EXTERN_C_END

#endif

