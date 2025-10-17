/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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
#import "config.h"
#import "WebCoreObjCExtras.h"

#import <utility>
#import <wtf/Assertions.h>
#import <wtf/MainThread.h>
#import <wtf/ObjCRuntimeExtras.h>
#import <wtf/Threading.h>

#if ASSERT_ENABLED

// This is like the isKindOfClass: method, bypassing it to get the correct answer for our purposes even for classes that override it.
// At the time of this writing, that included WebKit's WKObject class.
static bool safeIsKindOfClass(id object, Class testClass)
{
    if (!object)
        return false;
    for (auto ancestorClass = object_getClass(object); ancestorClass; ancestorClass = class_getSuperclass(ancestorClass)) {
        if (ancestorClass == testClass)
            return true;
    }
    return false;
}

#endif

bool WebCoreObjCScheduleDeallocateOnMainThread(Class deallocMethodClass, id object)
{
    ASSERT(safeIsKindOfClass(object, deallocMethodClass));

    if (isMainThread())
        return false;

    callOnMainThread([deallocMethodClass, object] {
        auto deallocSelector = sel_registerName("dealloc");
        wtfCallIMP<void>(method_getImplementation(class_getInstanceMethod(deallocMethodClass, deallocSelector)), object, deallocSelector);
    });

    return true;
}

bool WebCoreObjCScheduleDeallocateOnMainRunLoop(Class deallocMethodClass, id object)
{
    ASSERT(safeIsKindOfClass(object, deallocMethodClass));

    if (isMainRunLoop())
        return false;

    callOnMainRunLoop([deallocMethodClass, object] {
        auto deallocSelector = sel_registerName("dealloc");
        wtfCallIMP<void>(method_getImplementation(class_getInstanceMethod(deallocMethodClass, deallocSelector)), object, deallocSelector);
    });

    return true;
}
