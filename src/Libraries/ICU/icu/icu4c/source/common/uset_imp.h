/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
*   Copyright (C) 2004-2007, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
*******************************************************************************
*   file name:  uset_imp.h
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2004sep07
*   created by: Markus W. Scherer
*
*   Internal USet definitions.
*/

#ifndef __USET_IMP_H__
#define __USET_IMP_H__

#include "unicode/utypes.h"
#include "unicode/uset.h"

U_CDECL_BEGIN

typedef void U_CALLCONV
USetAdd(USet *set, UChar32 c);

typedef void U_CALLCONV
USetAddRange(USet *set, UChar32 start, UChar32 end);

typedef void U_CALLCONV
USetAddString(USet *set, const UChar *str, int32_t length);

typedef void U_CALLCONV
USetRemove(USet *set, UChar32 c);

typedef void U_CALLCONV
USetRemoveRange(USet *set, UChar32 start, UChar32 end);

/**
 * Interface for adding items to a USet, to keep low-level code from
 * statically depending on the USet implementation.
 * Calls will look like sa->add(sa->set, c);
 */
struct USetAdder {
    USet *set;
    USetAdd *add;
    USetAddRange *addRange;
    USetAddString *addString;
    USetRemove *remove;
    USetRemoveRange *removeRange;
};
typedef struct USetAdder USetAdder;

U_CDECL_END

#ifdef __cplusplus

namespace {

constexpr int32_t USET_CASE_MASK = USET_CASE_INSENSITIVE | USET_ADD_CASE_MAPPINGS;

}  // namespace

#endif  // __cplusplus

#endif
