/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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

// TEST_CONFIG

#include "test.h"
#include "testroot.i"

@interface One : TestRoot @end
@implementation One @end

@interface Two : TestRoot @end
@implementation Two @end

@interface Both : TestRoot @end
@implementation Both @end

@interface None : TestRoot @end
@implementation None @end


objc_hook_getImageName OnePreviousHook;
BOOL GetImageNameHookOne(Class cls, const char **outName)
{
    if (0 == strcmp(class_getName(cls), "One")) {
        *outName = "Image One";
        return YES;
    } else if (0 == strcmp(class_getName(cls), "Both")) {
        *outName = "Image Both via One";
        return YES;
    } else {
        return OnePreviousHook(cls, outName);
    }
}

objc_hook_getImageName TwoPreviousHook;
BOOL GetImageNameHookTwo(Class cls, const char **outName)
{
    if (0 == strcmp(class_getName(cls), "Two")) {
        *outName = "Image Two";
        return YES;
    } else if (0 == strcmp(class_getName(cls), "Both")) {
        *outName = "Image Both via Two";
        return YES;
    } else {
        return TwoPreviousHook(cls, outName);
    }
}

int main()
{

    // before hooks: main executable is the image name for four classes
    testassert(strstr(class_getImageName([One class]), "getImageNameHook"));
    testassert(strstr(class_getImageName([Two class]), "getImageNameHook"));
    testassert(strstr(class_getImageName([Both class]), "getImageNameHook"));
    testassert(strstr(class_getImageName([None class]), "getImageNameHook"));
    testassert(strstr(class_getImageName([NSObject class]), "libobjc"));

    // install hook One
    objc_setHook_getImageName(GetImageNameHookOne, &OnePreviousHook);

    // two classes are in Image One with hook One in place
    testassert(strstr(class_getImageName([One class]), "Image One"));
    testassert(strstr(class_getImageName([Two class]), "getImageNameHook"));
    testassert(strstr(class_getImageName([Both class]), "Image Both via One"));
    testassert(strstr(class_getImageName([None class]), "getImageNameHook"));
    testassert(strstr(class_getImageName([NSObject class]), "libobjc"));

    // install hook Two which chains to One
    objc_setHook_getImageName(GetImageNameHookTwo, &TwoPreviousHook);

    // two classes are in Image Two and one in One with both hooks in place
    testassert(strstr(class_getImageName([One class]), "Image One"));
    testassert(strstr(class_getImageName([Two class]), "Image Two"));
    testassert(strstr(class_getImageName([Both class]), "Image Both via Two"));
    testassert(strstr(class_getImageName([None class]), "getImageNameHook"));
    testassert(strstr(class_getImageName([NSObject class]), "libobjc"));

    succeed(__FILE__);
}
