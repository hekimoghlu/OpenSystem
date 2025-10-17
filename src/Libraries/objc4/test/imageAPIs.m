/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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

// TEST_CONFIG MEM=mrc

#include <objc/runtime.h>

#include "test.h"
#include "testroot.i"

@interface Foo: TestRoot @end
@implementation Foo @end
@interface Bar: Foo @end
@implementation Bar @end

int main(int argc __unused, const char **argv)
{
    // Make sure we show up in the list of image names.
    unsigned int count;
    const char **imageNames = objc_copyImageNames(&count);
    testassert(imageNames);
    testassertequal(imageNames[count], NULL);

    char myBaseName[MAXPATHLEN];
    basename_r(argv[0], myBaseName);
    const char *myImageName = NULL;
    for (unsigned int i = 0; i < count; i++) {
        char imageBaseName[MAXPATHLEN];
        basename_r(imageNames[i], imageBaseName);
        if (strcmp(imageBaseName, myBaseName) == 0)
            myImageName = imageNames[i];
    }
    testassert(myImageName);
    free(imageNames);

    // Make sure our classes show up in the class names.
    const char **classNames = objc_copyClassNamesForImage(myImageName, &count);
    testassert(classNames);
    testassertequal(classNames[count], NULL);

    int sawFoo = 0;
    int sawBar = 0;
    for (unsigned int i = 0; i < count; i++) {
        if (strcmp(classNames[i], "Foo") == 0)
            sawFoo++;
        if (strcmp(classNames[i], "Bar") == 0)
            sawBar++;
    }
    testassertequal(sawFoo, 1);
    testassertequal(sawBar, 1);
    free(classNames);

    // Make sure our classes show up in the classes list.
    sawFoo = 0;
    sawBar = 0;
    Class *classes = objc_copyClassesForImage(myImageName, &count);
    testassert(classes);
    testassertequal(classes[count], NULL);
    for (unsigned int i = 0; i < count; i++) {
        if (strcmp(class_getName(classes[i]), "Foo") == 0)
            sawFoo++;
        if (strcmp(class_getName(classes[i]), "Bar") == 0)
            sawBar++;
    }
    testassertequal(sawFoo, 1);
    testassertequal(sawBar, 1);
    free(classes);

    // Make sure bad names return NULL.
    testassertequal(objc_copyClassNamesForImage("aaaaaaaaaaaaaaaaaaa", NULL), NULL);
    testassertequal(objc_copyClassesForImage("aaaaaaaaaaaaaaaaaaa", NULL), NULL);

    succeed(__FILE__);
}
