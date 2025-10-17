/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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
*   Copyright (C) 1997-2012, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
******************************************************************************
*
*  FILE NAME : ptypes.h
*
*   Date        Name        Description
*   05/13/98    nos         Creation (content moved here from ptypes.h).
*   03/02/99    stephen     Added AS400 support.
*   03/30/99    stephen     Added Linux support.
*   04/13/99    stephen     Reworked for autoconf.
*   09/18/08    srl         Moved basic types back to ptypes.h from platform.h
******************************************************************************
*/

/**
 * \file
 * \brief C API: Definitions of integer types of various widths
 */

#ifndef _PTYPES_H
#define _PTYPES_H

/**
 * \def __STDC_LIMIT_MACROS
 * According to the Linux stdint.h, the ISO C99 standard specifies that in C++ implementations
 * macros like INT32_MIN and UINTPTR_MAX should only be defined if explicitly requested.
 * We need to define __STDC_LIMIT_MACROS before including stdint.h in C++ code
 * that uses such limit macros.
 * @internal
 */
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif

/* NULL, size_t, wchar_t */
#include <stddef.h>

/* More platform-specific definitions. */
#include "unicode/platform.h"

/*===========================================================================*/
/* Generic data types                                                        */
/*===========================================================================*/

#include <stdint.h>

// C++11 and C11 both specify that the data type char16_t should exist, C++11
// as a keyword and C11 as a typedef in the uchar.h header file, but not all
// implementations (looking at you, Apple, spring 2024) actually do this, so
// ICU4C must detect and deal with that.
#if !defined(__cplusplus) && !defined(U_IN_DOXYGEN)
#   if U_HAVE_CHAR16_T
#       include <uchar.h>
#   else
        typedef uint16_t char16_t;
#   endif
#endif

#endif /* _PTYPES_H */
