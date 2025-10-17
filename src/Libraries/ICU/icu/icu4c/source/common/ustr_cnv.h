/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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
*   Copyright (C) 1999-2010, International Business Machines
*   Corporation and others.  All Rights Reserved.
**********************************************************************
*   file name:  ustr_cnv.h
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2004Aug27
*   created by: George Rhoten
*/

#ifndef USTR_CNV_IMP_H
#define USTR_CNV_IMP_H

#include "unicode/utypes.h"
#include "unicode/ucnv.h"

#if !UCONFIG_NO_CONVERSION

/**
 * Get the default converter. This is a commonly used converter
 * that is used for the ustring and UnicodeString API.
 * Remember to use the u_releaseDefaultConverter when you are done.
 * @internal
 */
U_CAPI UConverter* U_EXPORT2
u_getDefaultConverter(UErrorCode *status);


/**
 * Release the default converter to the converter cache.
 * @internal
 */
U_CAPI void U_EXPORT2
u_releaseDefaultConverter(UConverter *converter);

/**
 * Flush the default converter, if cached. 
 * @internal
 */
U_CAPI void U_EXPORT2
u_flushDefaultConverter(void);

#endif

#endif
