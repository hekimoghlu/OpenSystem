/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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


#ifndef OPENSOURCE
#include <CoreFoundation/CoreFoundation.h>
#include <CoreFoundation/CFRuntime.h>
#endif

struct heim_base {
#ifndef OPENSOURCE
    CFRuntimeBase base;
#else
#ifdef HEIM_BASE_INTERNAL
    heim_type_t isa;
    heim_base_atomic_type ref_cnt;
    HEIM_TAILQ_ENTRY(heim_base) autorel;
    heim_auto_release_t autorelpool;
    uintptr_t isaextra[3];
#else
    void *data[8];
#endif
#endif
};

/* specialized version of base */
struct heim_base_uniq {
#ifndef OPENSOURCE
    CFRuntimeBase base;
    const char *name;
    void (*dealloc)(void *);
#else
#ifdef HEIM_BASE_INTERNAL
    heim_type_t isa;
    heim_base_atomic_type ref_cnt;
    HEIM_TAILQ_ENTRY(heim_base) autorel;
    heim_auto_release_t autorelpool;
    const char *name;
    void (*dealloc)(void *);
    uintptr_t isaextra[1];
#else
    void *data[8];
#endif
#endif
};

