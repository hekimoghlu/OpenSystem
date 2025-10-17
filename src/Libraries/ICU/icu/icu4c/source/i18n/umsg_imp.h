/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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
**********************************************************************
*   Copyright (C) 2001, International Business Machines
*   Corporation and others.  All Rights Reserved.
**********************************************************************
*   file name:  umsg_imp.h
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2001jun22
*   created by: George Rhoten
*/

#ifndef UMISC_H
#define UMISC_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

/* global variables used by the C and C++ message formatting API. */

extern const char16_t  *g_umsgTypeList[];
extern const char16_t  *g_umsgModifierList[];
extern const char16_t  *g_umsgDateModifierList[];
extern const int32_t g_umsgListLength;

extern const char16_t g_umsg_number[];
extern const char16_t g_umsg_date[];
extern const char16_t g_umsg_time[];
extern const char16_t g_umsg_choice[];

extern const char16_t g_umsg_currency[];
extern const char16_t g_umsg_percent[];
extern const char16_t g_umsg_integer[];

extern const char16_t g_umsg_short[];
extern const char16_t g_umsg_medium[];
extern const char16_t g_umsg_long[];
extern const char16_t g_umsg_full[];

#endif /* #if !UCONFIG_NO_FORMATTING */

#endif
