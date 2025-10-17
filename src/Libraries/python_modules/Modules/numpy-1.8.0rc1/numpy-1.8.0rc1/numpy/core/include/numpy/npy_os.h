/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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

#ifndef _NPY_OS_H_
#define _NPY_OS_H_

#if defined(linux) || defined(__linux) || defined(__linux__)
    #define NPY_OS_LINUX
#elif defined(__FreeBSD__) || defined(__NetBSD__) || \
            defined(__OpenBSD__) || defined(__DragonFly__)
    #define NPY_OS_BSD
    #ifdef __FreeBSD__
        #define NPY_OS_FREEBSD
    #elif defined(__NetBSD__)
        #define NPY_OS_NETBSD
    #elif defined(__OpenBSD__)
        #define NPY_OS_OPENBSD
    #elif defined(__DragonFly__)
        #define NPY_OS_DRAGONFLY
    #endif
#elif defined(sun) || defined(__sun)
    #define NPY_OS_SOLARIS
#elif defined(__CYGWIN__)
    #define NPY_OS_CYGWIN
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    #define NPY_OS_WIN32
#elif defined(__APPLE__)
    #define NPY_OS_DARWIN
#else
    #define NPY_OS_UNKNOWN
#endif

#endif
