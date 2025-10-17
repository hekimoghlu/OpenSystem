/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
*   Copyright (C) 1998-2016, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
*******************************************************************************
*
* File error.h
*
* Modification History:
*
*   Date        Name        Description
*   05/28/99    stephen     Creation.
*******************************************************************************
*/

#ifndef ERROR_H
#define ERROR_H 1

#include "unicode/utypes.h"

U_CDECL_BEGIN

extern const char *gCurrentFileName;

#if APPLE_ICU_CHANGES
// rdar://
__attribute__((format(printf, 2, 3)))
#endif  // APPLE_ICU_CHANGES
U_CFUNC void error(uint32_t linenumber, const char *msg, ...);
#if APPLE_ICU_CHANGES
// rdar://
__attribute__((format(printf, 2, 3)))
#endif  // APPLE_ICU_CHANGES
U_CFUNC void warning(uint32_t linenumber, const char *msg, ...);

/* Show warnings? */
U_CFUNC void setShowWarning(UBool val);
U_CFUNC UBool getShowWarning(void);

/* strict */
U_CFUNC void setStrict(UBool val);
U_CFUNC UBool isStrict(void);

/* verbosity */
U_CFUNC void setVerbose(UBool val);
U_CFUNC UBool isVerbose(void);

U_CDECL_END

#endif
