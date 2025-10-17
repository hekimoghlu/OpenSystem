/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 22, 2024.
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
*                                                                            *
* Copyright (C) 2001-2011, International Business Machines                   *
*                Corporation and others. All Rights Reserved.                *
*                                                                            *
******************************************************************************
*   file name:  ucln_io.h
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2006August11
*   created by: George Rhoten
*/

#ifndef __UCLN_IO_H__
#define __UCLN_IO_H__

#include "unicode/utypes.h"
#include "ucln.h"

/*
Please keep the order of enums declared in same order
as the functions are suppose to be called. */
typedef enum ECleanupIOType {
    UCLN_IO_START = -1,
    UCLN_IO_LOCBUND,
    UCLN_IO_PRINTF,
    UCLN_IO_COUNT /* This must be last */
} ECleanupIOType;

/* Main library cleanup registration function. */
/* See common/ucln.h for details on adding a cleanup function. */
U_CFUNC void U_EXPORT2 ucln_io_registerCleanup(ECleanupIOType type,
                                                 cleanupFunc *func);

#endif
