/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
/* Use this header for Carbon on OSX */
/* --das */

#define TARGET_API_MAC_CARBON 1
#define TARGET_API_MAC_OSX 1
#define TCLAE_OSX 1 //  das 260601
#define TCLAE_CARBON 1

#ifdef __APPLE_CC__ //  das 260601
#define TCLAE_GCC 1
#endif

#ifdef TCLAE_GCC //  das 260601
//  das 260601 GCC will actually include the precompiled header <Carbon/Carbon.p> if present
#ifdef TCLAE_USE_FRAMEWORK_INCLUDES
#include <Carbon/Carbon.h>
#else
#include <ConditionalMacros.h>
#endif
//  das 260601 dummy typedefs for old types, remove when no longer needed
struct SFReply {};
typedef struct SFReply SFReply;
typedef struct SFReply StandardFileReply;
#else
#include <ConditionalMacros.h>
#endif //TCLAE_GCC

#if TARGET_RT_MAC_MACHO //  das 260601 compiling natively on OSX, either with GCC or a new CW
#define TCLAE_MACHO 1
#endif

#ifdef TCLAE_MACHO //  das 260601
#define TCLAE_PATH_SEP '/'
#else
#define TCLAE_PATH_SEP ':'
#endif

#include <tcl.h>

#ifndef CONST84 // Tcl 8.4 backwards compatibility
#      define CONST84 
#      define CONST84_RETURN CONST
#endif

#include "tclAE.h"
