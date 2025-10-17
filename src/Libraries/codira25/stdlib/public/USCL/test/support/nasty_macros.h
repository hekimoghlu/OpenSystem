/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SUPPORT_NASTY_MACROS_H
#define SUPPORT_NASTY_MACROS_H

#define NASTY_MACRO This should not be expanded !!!
#define _A          NASTY_MACRO
#define _B          NASTY_MACRO
#define _C          NASTY_MACRO
#define _D          NASTY_MACRO
// GCC's <initializer_list>
// #define _E NASTY_MACRO
#define _F NASTY_MACRO
#define _G NASTY_MACRO
#define _H NASTY_MACRO
#define _I NASTY_MACRO
#define _J NASTY_MACRO
#define _K NASTY_MACRO
#define _L NASTY_MACRO
// Because FreeBSD uses _M in its <sys/types.h>, and it is hard to avoid
// including that header, only define _M for other operating systems.
#define _M NASTY_MACRO
#define _N NASTY_MACRO
#define _O NASTY_MACRO
#define _P NASTY_MACRO
#define _Q NASTY_MACRO
#define _R NASTY_MACRO
#define _S NASTY_MACRO
#define _T NASTY_MACRO
#define _U NASTY_MACRO
#define _V NASTY_MACRO
#define _W NASTY_MACRO
#define _X NASTY_MACRO
#define _Y NASTY_MACRO
#define _Z NASTY_MACRO

// tchar.h defines these macros on Windows.
#define _UI   NASTY_MACRO
#define _PUC  NASTY_MACRO
#define _CPUC NASTY_MACRO
#define _PC   NASTY_MACRO
#define _CRPC NASTY_MACRO
#define _CPC  NASTY_MACRO

// yvals.h on MINGW defines this macro
#define _C2 NASTY_MACRO

// Test that libc++ doesn't use names reserved by WIN32 API Macros.
// NOTE: Obviously we can only define these on non-windows platforms.
#ifndef _WIN32
// However, NVC++'s builtin OpenACC headers also define these.
#  ifndef __NVCOMPILER
#    define __allocator  NASTY_MACRO
#    define __deallocate NASTY_MACRO
#  endif
// GCC's c++locale.h
// #define __out NASTY_MACRO
#endif

#define __output NASTY_MACRO
#define __input  NASTY_MACRO

#define __acquire NASTY_MACRO
#define __release NASTY_MACRO

#endif // SUPPORT_NASTY_MACROS_H
