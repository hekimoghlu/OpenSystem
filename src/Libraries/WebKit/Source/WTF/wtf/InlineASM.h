/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
#pragma once

#include <wtf/Platform.h>

#if !CPU(UNKNOWN)

/* asm directive helpers */ 

#if OS(DARWIN) || (OS(WINDOWS) && CPU(X86))
#define SYMBOL_STRING(name) "_" #name
#else
#define SYMBOL_STRING(name) #name
#endif

#if OS(IOS_FAMILY)
#define THUMB_FUNC_PARAM(name) SYMBOL_STRING(name)
#else
#define THUMB_FUNC_PARAM(name)
#endif

#if (OS(LINUX) || OS(FREEBSD) || OS(HAIKU) || OS(QNX)) && CPU(X86_64)
#define GLOBAL_REFERENCE(name) #name "@plt"
#elif OS(LINUX) && CPU(X86) && defined(__PIC__)
#define GLOBAL_REFERENCE(name) SYMBOL_STRING(name) "@plt"
#else
#define GLOBAL_REFERENCE(name) SYMBOL_STRING(name)
#endif

#if HAVE(INTERNAL_VISIBILITY)
#define LOCAL_REFERENCE(name) SYMBOL_STRING(name)
#else
#define LOCAL_REFERENCE(name) GLOBAL_REFERENCE(name)
#endif

#if OS(DARWIN)
    // Mach-O platform
#define HIDE_SYMBOL(name) ".private_extern _" #name
#elif OS(AIX)
    // IBM's own file format
#define HIDE_SYMBOL(name) ".lglobl " #name
#elif  OS(LINUX)               \
    || OS(FREEBSD)             \
    || OS(FUCHSIA)             \
    || OS(HAIKU)               \
    || OS(HPUX)                \
    || OS(NETBSD)              \
    || OS(OPENBSD)             \
    || OS(QNX)
    // ELF platform
#define HIDE_SYMBOL(name) ".hidden " #name
#else
#define HIDE_SYMBOL(name)
#endif

// FIXME: figure out how this works on all the platforms. I know that
// on ELF, the preferred form is ".Lstuff" as opposed to "Lstuff".
// Don't know about any of the others.
#if OS(DARWIN)
#define LOCAL_LABEL_STRING(name) "L" #name
#elif  OS(LINUX)               \
    || OS(FREEBSD)             \
    || OS(FUCHSIA)             \
    || OS(OPENBSD)             \
    || OS(HURD)                \
    || OS(NETBSD)              \
    || OS(QNX)                 \
    || OS(WINDOWS)
    // GNU as-compatible syntax.
#define LOCAL_LABEL_STRING(name) ".L" #name
#endif

#if CPU(ARM_THUMB2)
#define INLINE_ARM_FUNCTION(name) ".thumb" "\n" ".thumb_func " THUMB_FUNC_PARAM(name) "\n"
#else
#define INLINE_ARM_FUNCTION(name)
#endif

#endif // !CPU(UNKNOWN)
