/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
#include "test.h"
#include <dlfcn.h>
#include <objc/objc-internal.h>

#define ADD_IMAGE_CALLBACK(n)                                                  \
int called ## n = 0;                                                           \
static void add_image ## n(const struct mach_header * _Nonnull mh __unused,    \
                           struct _dyld_section_location_info_s *_Nonnull      \
                           info __unused) {                                    \
    called ## n++;                                                             \
}

ADD_IMAGE_CALLBACK(1)
ADD_IMAGE_CALLBACK(2)
ADD_IMAGE_CALLBACK(3)
ADD_IMAGE_CALLBACK(4)
ADD_IMAGE_CALLBACK(5)

int main()
{
    objc_addLoadImageFunc2(&add_image1);
    testassert(called1 > 0);
    int oldcalled = called1;
    void *handle = dlopen("load-image-notification1.dylib", RTLD_LAZY);
    testassert(handle);
    testassert(called1 > oldcalled);

    objc_addLoadImageFunc2(add_image2);
    testassert(called2 == called1);
    oldcalled = called1;
    handle = dlopen("load-image-notification2.dylib", RTLD_LAZY);
    testassert(handle);
    testassert(called1 > oldcalled);
    testassert(called2 == called1);

    objc_addLoadImageFunc2(add_image3);
    testassert(called3 == called1);
    oldcalled = called1;
    handle = dlopen("load-image-notification3.dylib", RTLD_LAZY);
    testassert(handle);
    testassert(called1 > oldcalled);
    testassert(called2 == called1);
    testassert(called3 == called1);

    objc_addLoadImageFunc2(add_image4);
    testassert(called4 == called1);
    oldcalled = called1;
    handle = dlopen("load-image-notification4.dylib", RTLD_LAZY);
    testassert(handle);
    testassert(called1 > oldcalled);
    testassert(called2 == called1);
    testassert(called3 == called1);
    testassert(called4 == called1);

    objc_addLoadImageFunc2(add_image5);
    testassert(called5 == called1);
    oldcalled = called1;
    handle = dlopen("load-image-notification5.dylib", RTLD_LAZY);
    testassert(handle);
    testassert(called1 > oldcalled);
    testassert(called2 == called1);
    testassert(called3 == called1);
    testassert(called4 == called1);
    testassert(called5 == called1);

    succeed(__FILE__);
}
