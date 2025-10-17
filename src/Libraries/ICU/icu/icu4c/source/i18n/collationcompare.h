/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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
* Copyright (C) 1996-2014, International Business Machines
* Corporation and others.  All Rights Reserved.
*******************************************************************************
* collationcompare.h
*
* created on: 2012feb14 with new and old collation code
* created by: Markus W. Scherer
*/

#ifndef __COLLATIONCOMPARE_H__
#define __COLLATIONCOMPARE_H__

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "unicode/ucol.h"

U_NAMESPACE_BEGIN

class CollationIterator;
struct CollationSettings;

class U_I18N_API CollationCompare /* not : public UObject because all methods are static */ {
public:
    static UCollationResult compareUpToQuaternary(CollationIterator &left, CollationIterator &right,
                                                  const CollationSettings &settings,
                                                  UErrorCode &errorCode);
};

U_NAMESPACE_END

#endif  // !UCONFIG_NO_COLLATION
#endif  // __COLLATIONCOMPARE_H__
