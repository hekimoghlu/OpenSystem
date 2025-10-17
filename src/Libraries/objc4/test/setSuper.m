/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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
#include <objc/runtime.h>

@interface Super1 : TestRoot @end
@implementation Super1
+(int)classMethod { return 1; }
-(int)instanceMethod { return 10000; }
@end

@interface Super2 : TestRoot @end
@implementation Super2
+(int)classMethod { return 2; }
-(int)instanceMethod { return 20000; }
@end

@interface Sub : Super1 @end
@implementation Sub
+(int)classMethod { return [super classMethod] + 100; }
-(int)instanceMethod { 
    return [super instanceMethod] + 1000000;
}
@end

int main()
{
    Class cls;
    Sub *obj = [Sub new];

    testassert(101 == [[Sub class] classMethod]);
    testassert(1010000 == [obj instanceMethod]);

    cls = class_setSuperclass([Sub class], [Super2 class]);

    testassert(cls == [Super1 class]);
    testassert(object_getClass(cls) == object_getClass([Super1 class]));

    testassert(102 == [[Sub class] classMethod]);
    testassert(1020000 == [obj instanceMethod]);

    succeed(__FILE__);
}
