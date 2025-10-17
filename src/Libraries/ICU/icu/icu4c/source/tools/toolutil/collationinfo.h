/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
* Copyright (C) 2013-2015, International Business Machines
* Corporation and others.  All Rights Reserved.
*******************************************************************************
* collationinfo.h
*
* created on: 2013aug05
* created by: Markus W. Scherer
*/

#ifndef __COLLATIONINFO_H__
#define __COLLATIONINFO_H__

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

U_NAMESPACE_BEGIN

struct CollationData;

/**
 * Collation-related code for tools & demos.
 */
class U_TOOLUTIL_API CollationInfo /* all static */ {
public:
    static void printSizes(int32_t sizeWithHeader, const int32_t indexes[]);
    static void printReorderRanges(const CollationData &data, const int32_t *codes, int32_t length);

private:
    CollationInfo();  // no constructor

    static int32_t getDataLength(const int32_t indexes[], int32_t startIndex);
};

U_NAMESPACE_END

#endif  // !UCONFIG_NO_COLLATION
#endif  // __COLLATIONINFO_H__
