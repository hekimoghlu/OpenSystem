/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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
*   Copyright (C) 2002-2011, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
******************************************************************************
*
* File uassert.h
*
*  Contains the U_ASSERT and UPRV_UNREACHABLE_* macros
*
******************************************************************************
*/
#ifndef U_ASSERT_H
#define U_ASSERT_H

/* utypes.h is included to get the proper define for uint8_t */
#include "unicode/utypes.h"
/* for abort */
#include <stdlib.h>

/**
 * \def U_ASSERT
 * By default, U_ASSERT just wraps the C library assert macro.
 * By changing the definition here, the assert behavior for ICU can be changed
 * without affecting other non - ICU uses of the C library assert().
*/
#if U_DEBUG
#   include <assert.h>
#   define U_ASSERT(exp) assert(exp)
#elif U_CPLUSPLUS_VERSION
#   define U_ASSERT(exp) (void)0
#else
#   define U_ASSERT(exp)
#endif

/**
 * \def UPRV_UNREACHABLE_ASSERT
 * This macro is used in places that we had believed were unreachable, but
 * experience has shown otherwise (possibly due to memory corruption, etc).
 * In this case we call assert() in debug versions as with U_ASSERT, instead
 * of unconditionally calling abort(). However we also allow redefinition as
 * with UPRV_UNREACHABLE_EXIT.
 * @internal
*/
#if defined(UPRV_UNREACHABLE_ASSERT)
    // Use the predefined value.
#elif U_DEBUG
#   include <assert.h>
#   define UPRV_UNREACHABLE_ASSERT assert(false)
#elif U_CPLUSPLUS_VERSION
#   define UPRV_UNREACHABLE_ASSERT (void)0
#else
#   define UPRV_UNREACHABLE_ASSERT
#endif

/**
 * \def UPRV_UNREACHABLE_EXIT
 * This macro is used to unconditionally abort if unreachable code is ever executed.
 * @internal
*/
#if defined(UPRV_UNREACHABLE_EXIT)
    // Use the predefined value.
#else
#   define UPRV_UNREACHABLE_EXIT abort()
#endif

#endif
