/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 13, 2025.
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
********************************************************************************
*   Copyright (C) 2008-2011, International Business Machines
*   Corporation and others.  All Rights Reserved.
********************************************************************************
*
* File WINTZIMPL.H
*
********************************************************************************
*/

#ifndef __WINTZIMPL
#define __WINTZIMPL

#include "unicode/utypes.h"

#if U_PLATFORM_USES_ONLY_WIN32_API
/**
 * \file 
 * \brief C API: Utilities for dealing w/ Windows time zones.
 */
U_CDECL_BEGIN
/* Forward declarations for Windows types... */
typedef struct _TIME_ZONE_INFORMATION TIME_ZONE_INFORMATION;
U_CDECL_END

/*
 * This method was moved over from common/wintz.h to allow for access to i18n functions
 * needed to get the Windows time zone information without using static tables.
 */
U_CAPI UBool U_EXPORT2
uprv_getWindowsTimeZoneInfo(TIME_ZONE_INFORMATION *zoneInfo, const UChar *icuid, int32_t length);


#endif /* U_PLATFORM_USES_ONLY_WIN32_API */

#endif /* __WINTZIMPL */
