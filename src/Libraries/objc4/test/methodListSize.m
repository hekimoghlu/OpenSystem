/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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
// rdar://8052003 rdar://8077031

#include "test.h"

#include <malloc/malloc.h>
#include <objc/runtime.h>

// add SELCOUNT methods to each of CLASSCOUNT classes
#define CLASSCOUNT 100
#define SELCOUNT 200

int main()
{
    int i, j;
    malloc_statistics_t start, end;

    Class root;
    root = objc_allocateClassPair(NULL, "Root", 0);
    objc_registerClassPair(root);

    Class classes[CLASSCOUNT];
    for (i = 0; i < CLASSCOUNT; i++) {
        char *classname;
        asprintf(&classname, "GrP_class_%d", i);
        classes[i] = objc_allocateClassPair(root, classname, 0);
        objc_registerClassPair(classes[i]);
        free(classname);
    }

    SEL selectors[SELCOUNT];
    for (i = 0; i < SELCOUNT; i++) {
        char *selname;
        asprintf(&selname, "GrP_sel_%d", i);
        selectors[i] = sel_registerName(selname);
        free(selname);
    }

    malloc_zone_statistics(NULL, &start);

    for (i = 0; i < CLASSCOUNT; i++) {
        for (j = 0; j < SELCOUNT; j++) {
            class_addMethod(classes[i], selectors[j], (IMP)main, "");
        }
    }

    malloc_zone_statistics(NULL, &end);

    // expected: 3-word method struct plus two other words
    ssize_t expected = (sizeof(void*) * (3+2)) * SELCOUNT * CLASSCOUNT;
    ssize_t actual = end.size_in_use - start.size_in_use;
    testassert(actual < expected * 3);  // allow generous fudge factor

    succeed(__FILE__);
}

