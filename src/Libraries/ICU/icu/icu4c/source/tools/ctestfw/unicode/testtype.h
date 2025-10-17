/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 12, 2024.
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
 *****************************************************************************************
 *   Copyright (C) 2004-2011, International Business Machines
 *   Corporation and others.  All Rights Reserved.
 *****************************************************************************************
 */

#include "unicode/utypes.h"

/*Deals with imports and exports of the dynamic library*/
#if !defined(U_STATIC_IMPLEMENTATION)
    #define T_CTEST_EXPORT U_EXPORT
    #define T_CTEST_IMPORT U_IMPORT
#else
    #define T_CTEST_EXPORT
    #define T_CTEST_IMPORT
#endif

#if defined(_MSC_VER)
#define T_CTEST_EXPORT2 __cdecl
#else
#define T_CTEST_EXPORT2
#endif

#ifdef __cplusplus
    #define C_CTEST_API extern "C"
    U_NAMESPACE_USE
#else
    #define C_CTEST_API
#endif

#ifdef T_CTEST_IMPLEMENTATION
    #define T_CTEST_API C_CTEST_API  T_CTEST_EXPORT
    #define T_CTEST_EXPORT_API T_CTEST_EXPORT
#else
    #define T_CTEST_API C_CTEST_API  T_CTEST_IMPORT
    #define T_CTEST_EXPORT_API T_CTEST_IMPORT
#endif
