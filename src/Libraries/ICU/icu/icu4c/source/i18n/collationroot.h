/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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
* Copyright (C) 2012-2014, International Business Machines
* Corporation and others.  All Rights Reserved.
*******************************************************************************
* collationroot.h
*
* created on: 2012dec17
* created by: Markus W. Scherer
*/

#ifndef __COLLATIONROOT_H__
#define __COLLATIONROOT_H__

#include "unicode/utypes.h"
#include "unicode/udata.h"

#if !UCONFIG_NO_COLLATION

U_NAMESPACE_BEGIN

struct CollationCacheEntry;
struct CollationData;
struct CollationSettings;
struct CollationTailoring;

/**
 * Collation root provider.
 */
class U_I18N_API CollationRoot {  // purely static
public:
    static const CollationCacheEntry *getRootCacheEntry(UErrorCode &errorCode);
    static const CollationTailoring *getRoot(UErrorCode &errorCode);
    static const CollationData *getData(UErrorCode &errorCode);
    static const CollationSettings *getSettings(UErrorCode &errorCode);
    static void U_EXPORT2 forceLoadFromFile(const char* ucadataPath, UErrorCode &errorCode);

private:
    static void U_CALLCONV load(const char* ucadataPath, UErrorCode &errorCode);
    static UDataMemory* loadFromFile(const char* ucadataPath, UErrorCode &errorCode);
};

U_NAMESPACE_END

#endif  // !UCONFIG_NO_COLLATION
#endif  // __COLLATIONROOT_H__
