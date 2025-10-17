/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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
******************************************************************************
*
*   Copyright (C) 1996-2013, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
******************************************************************************
*
* File locmap.h      : Locale Mapping Classes
* 
*
* Created by: Helena Shih
*
* Modification History:
*
*  Date        Name        Description
*  3/11/97     aliu        Added setId().
*  4/20/99     Madhu       Added T_convertToPosix()
* 09/18/00     george      Removed the memory leaks.
* 08/23/01     george      Convert to C
*============================================================================
*/

#ifndef LOCMAP_H
#define LOCMAP_H

#include "unicode/utypes.h"

#define LANGUAGE_LCID(hostID) (uint16_t)(0x03FF & hostID)

U_CAPI int32_t uprv_convertToPosix(uint32_t hostid, char* posixID, int32_t posixIDCapacity, UErrorCode* status);

/* Don't call these functions directly. Use uloc_getLCID instead. */
U_CAPI uint32_t uprv_convertToLCIDPlatform(const char* localeID, UErrorCode* status); // Leverage platform conversion if possible
U_CAPI uint32_t uprv_convertToLCID(const char* langID, const char* posixID, UErrorCode* status);

#endif /* LOCMAP_H */

