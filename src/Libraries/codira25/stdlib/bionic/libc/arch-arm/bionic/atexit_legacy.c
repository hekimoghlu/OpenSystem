/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 21, 2024.
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
#include <sys/types.h>
#include <stdio.h>

#include <async_safe/log.h>

/*
 * This source file should only be included by libc.so, its purpose is
 * to support legacy ARM binaries by exporting a publicly visible
 * implementation of atexit().
 */

extern int __cxa_atexit(void (*func)(void *), void *arg, void *dso);

/*
 * Register a function to be performed at exit.
 */
int
atexit(void (*func)(void))
{
    /*
     * Exit functions queued by this version of atexit will not be called
     * on dlclose(), and when they are called (at program exit), the
     * calling library may have been dlclose()'d, causing the program to
     * crash.
     */
    static char const warning[] = "WARNING: generic atexit() called from legacy shared library\n";

    async_safe_format_log(ANDROID_LOG_WARN, "libc", warning);
    fprintf(stderr, warning);

    return (__cxa_atexit((void (*)(void *))func, NULL, NULL));
}
