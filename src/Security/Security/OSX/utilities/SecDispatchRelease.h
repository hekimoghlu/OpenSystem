/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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
#ifndef _SECDISPATCHRELEASE_H_
#define _SECDISPATCHRELEASE_H_

#include <dispatch/dispatch.h>
#include <xpc/xpc.h>

#define dispatch_retain_safe(DO) {  \
    __typeof__(DO) _do = (DO);      \
    if (_do)                        \
        dispatch_retain(_do);       \
}

#define dispatch_release_safe(DO) { \
    __typeof__(DO) _do = (DO);      \
    if (_do)                        \
        dispatch_release(_do);      \
}

#define dispatch_release_null(DO) { \
    __typeof__(DO) _do = (DO);      \
    if (_do) {                      \
        (DO) = NULL;                \
        dispatch_release(_do);      \
    }                               \
}


#define xpc_retain_safe(XO) {  \
    __typeof__(XO) _xo = (XO); \
    if (_xo)                   \
        xpc_retain(_xo);       \
}

#define xpc_release_safe(XO) { \
    __typeof__(XO) _xo = (XO); \
    if (_xo)                   \
        xpc_release(_xo);      \
}

#define xpc_release_null(XO) {  \
    __typeof__(XO) _xo = (XO); \
    if (_xo) {                  \
        (XO) = NULL;            \
        xpc_release(_xo);       \
    }                           \
}

#endif

