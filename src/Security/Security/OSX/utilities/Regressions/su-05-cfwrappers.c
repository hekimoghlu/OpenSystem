/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 8, 2024.
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
#include <utilities/SecCFRelease.h>
#include <utilities/SecCFWrappers.h>
#include <utilities/SecIOFormat.h>

#include "utilities_regressions.h"

#define kCFWrappersTestCount 25

static CFDataRef *testCopyDataPtr(void) {
    static CFDataRef sData = NULL;
    if (!sData)
        sData = CFDataCreate(kCFAllocatorDefault, NULL, 0);
    else
        CFRetain(sData);
    return &sData;
}

static void
test_object(CFDataRef data) {
    CFDataRef myData = CFRetainSafe(data);
    ok(CFEqual(myData, data), "");
    is(CFGetRetainCount(myData), 2, "");
    ok(CFReleaseNull(myData) == ((CFDataRef)(0)), "");
    is(myData, NULL, "");

    is(CFGetRetainCount(data), 1);
    CFRetainAssign(myData, data);
    is(CFGetRetainCount(data), 2);
    CFRetainAssign(myData, data);
    is(CFGetRetainCount(data), 2);
    CFRetainAssign(myData, NULL);
    is(CFGetRetainCount(data), 1);
    is(myData, NULL, "");

    CFDataRef *pData = testCopyDataPtr();
    is(CFGetRetainCount(*pData), 1);
    CFDataRef objects[10] = {}, *object = objects;
    *object = *pData;
    CFRetainAssign(*testCopyDataPtr(), *object++);
    is(CFGetRetainCount(*pData), 2, "CFRetainAssign evaluates it's first argument argument %" PRIdCFIndex " times", CFGetRetainCount(*pData) - 1);
    is(object - objects, 1, "CFRetainAssign evaluates it's second argument %td times", object - objects);

    is(CFGetRetainCount(data), 1);
    CFAssignRetained(myData, data);
    is(CFGetRetainCount(myData), 1);
}

static void
test_null(void) {
    CFTypeRef nullObject1 = NULL;
    CFTypeRef nullObject2 = NULL;

    nullObject1 = CFRetainSafe(NULL);

    is(nullObject1, NULL, "");
    is(CFReleaseNull(nullObject1), NULL, "CFReleaseNull(nullObject1) returned");
    is(nullObject1, NULL);
    is(CFReleaseSafe(nullObject1), NULL, "CFReleaseSafe(nullObject1) returned");
    is(CFReleaseSafe(NULL), NULL, "CFReleaseSafe(NULL)");
    is(CFReleaseNull(nullObject2), NULL, "CFReleaseNull(nullObject2) returned");
    is(nullObject2, NULL, "nullObject2 still NULL");

    CFRetainAssign(nullObject2, nullObject1);

    CFTypeRef *object, objects[10] = {};

    object = &objects[0];
    CFRetainSafe(*object++);
    is(object - objects, 1, "CFRetainSafe evaluates it's argument %td times", object - objects);

    object = &objects[0];
    CFReleaseSafe(*object++);
    is(object - objects, 1, "CFReleaseSafe evaluates it's argument %td times", object - objects);

    object = &objects[0];
    CFReleaseNull(*object++);
    is(object - objects, 1, "CFReleaseNull evaluates it's argument %td times", object - objects);
}

int
su_05_cfwrappers(int argc, char *const *argv) {
    plan_tests(kCFWrappersTestCount);

    test_null();
    CFDataRef data = CFDataCreate(kCFAllocatorDefault, NULL, 0);
    test_object(data);
    CFReleaseNull(data);
    ok(data == NULL, "data is NULL now");
    return 0;
}
