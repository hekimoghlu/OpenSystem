/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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
* Copyright (C) 2013-2014, International Business Machines
* Corporation and others.  All Rights Reserved.
*******************************************************************************
* collationdatawriter.h
*
* created on: 2013aug06
* created by: Markus W. Scherer
*/

#ifndef __COLLATIONDATAWRITER_H__
#define __COLLATIONDATAWRITER_H__

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

U_NAMESPACE_BEGIN

struct CollationData;
struct CollationSettings;
struct CollationTailoring;

/**
 * Collation-related code for tools & demos.
 */
class U_I18N_API CollationDataWriter /* all static */ {
public:
    static int32_t writeBase(const CollationData &data, const CollationSettings &settings,
                             const void *rootElements, int32_t rootElementsLength,
                             int32_t indexes[], uint8_t *dest, int32_t capacity,
                             UErrorCode &errorCode);

    static int32_t writeTailoring(const CollationTailoring &t, const CollationSettings &settings,
                                  int32_t indexes[], uint8_t *dest, int32_t capacity,
                                  UErrorCode &errorCode);

private:
    CollationDataWriter() = delete;  // no constructor

    static int32_t write(UBool isBase, const UVersionInfo dataVersion,
                         const CollationData &data, const CollationSettings &settings,
                         const void *rootElements, int32_t rootElementsLength,
                         int32_t indexes[], uint8_t *dest, int32_t capacity,
                         UErrorCode &errorCode);

    static void copyData(const int32_t indexes[], int32_t startIndex,
                         const void *src, uint8_t *dest);
};

U_NAMESPACE_END

#endif  // !UCONFIG_NO_COLLATION
#endif  // __COLLATIONDATAWRITER_H__
