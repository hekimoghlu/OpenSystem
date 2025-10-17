/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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

// TEST_CONFIG OS=!exclavekit

#include "test.h"
#include <mach/mach.h>
#include <malloc/malloc.h>

// Look for malloc zone "ObjC" iff OBJC_USE_INTERNAL_ZONE is set.
// This fails if objc tries to allocate before checking its own 
// environment variables (rdar://6688423)

int main()
{
    if (is_guardmalloc()) {
        // guard malloc confuses this test
        succeed(__FILE__);
    }

    kern_return_t kr;
    vm_address_t *zones;
    unsigned int count, i;
    BOOL has_objc = NO, want_objc = NO;

    want_objc = (getenv("OBJC_USE_INTERNAL_ZONE") != NULL) ? YES : NO;
    testprintf("want objc %s\n", want_objc ? "YES" : "NO");

    kr = malloc_get_all_zones(mach_task_self(), NULL, &zones, &count);
    testassert(!kr);
    for (i = 0; i < count; i++) {
        const char *name = malloc_get_zone_name((malloc_zone_t *)zones[i]);
        if (name) {
            BOOL is_objc = (0 == strcmp(name, "ObjC_Internal")) ? YES : NO;
            if (is_objc) has_objc = YES;
            testprintf("zone %s\n", name);
        }
    }

    testassert(want_objc == has_objc);

    succeed(__FILE__);
}
