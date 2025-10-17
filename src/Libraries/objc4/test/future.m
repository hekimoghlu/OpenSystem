/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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
// NOTE: ld-prime now eliminates classrefs, which breaks future classes
// somewhat. We don't plan to fix this for now, but we do want to keep older
// binaries working. We build future2.dylib with -ld_classic to force it to
// still use classrefs, which allows this test to verify that future classes
// still work in that case.

#include "test.h"

#if __has_feature(objc_arc)

int main()
{
    testwarn("rdar://10041403 future class API is not ARC-compatible");
    succeed(__FILE__);
}


#else

#include <objc/runtime.h>
#include <malloc/malloc.h>
#include <string.h>
#include <dlfcn.h>
#include "future.h"

@implementation Sub2 
+(int)method { 
    return 2;
}
+(Class)classref {
    return [Sub2 class];
}
@end

@implementation SubSub2
+(int)method {
    return 1 + [super method];
}
@end

int main()
{
    Class oldTestRoot;
    Class oldSub1;

    // objc_getFutureClass with existing class
    oldTestRoot = objc_getFutureClass("TestRoot");
    testassert(oldTestRoot == [TestRoot class]);
    testassert(! _class_isFutureClass(oldTestRoot));

    // objc_getFutureClass with missing class
    oldSub1 = objc_getFutureClass("Sub1");
    testassert(oldSub1);
    testassert(malloc_size((__bridge void*)oldSub1) > 0);
    testassert(objc_getClass("Sub1") == Nil);
    testassert(_class_isFutureClass(oldSub1));
    testassert(0 == strcmp(class_getName(oldSub1), "Sub1"));
    testassert(object_getClass(oldSub1) == Nil);  // CF expects this

    // objc_getFutureClass a second time
    testassert(oldSub1 == objc_getFutureClass("Sub1"));

#if !TARGET_OS_EXCLAVEKIT
    // Load class Sub1
    dlopen("future2.dylib", 0);

    // Verify use of future class
    Class newSub1 = objc_getClass("Sub1");
    testassert(oldSub1 == newSub1);
    testassert(newSub1 == [newSub1 classref]);
    testassert(newSub1 == class_getSuperclass(objc_getClass("SubSub1")));
    testassert(! _class_isFutureClass(newSub1));

    testassert(1 == [oldSub1 method]);
    testassert(1 == [newSub1 method]);
#endif // !TARGET_OS_EXCLAVEKIT

    succeed(__FILE__);
}

#endif
