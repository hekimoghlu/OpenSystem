/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
#include <objc/runtime.h>
#include <objc/objc-internal.h>
#import <objc/NSObject.h>

class SerialNumber {
    size_t _number;
public:
    SerialNumber() : _number(42) {}
    SerialNumber(const SerialNumber &number) : _number(number._number + 1) {}
    SerialNumber &operator=(const SerialNumber &number) { _number = number._number + 1; return *this; }

    int operator==(const SerialNumber &number) { return _number == number._number; }
    int operator!=(const SerialNumber &number) { return _number != number._number; }
};

@interface TestAtomicProperty : NSObject {
    SerialNumber number;
}
@property(atomic) SerialNumber number;
@end

@implementation TestAtomicProperty

@synthesize number;

@end

int main()
{
    PUSH_POOL {
        SerialNumber number;
        TestAtomicProperty *test = [TestAtomicProperty new];
        test.number = number;
        testassert(test.number != number);
    } POP_POOL;

    succeed(__FILE__);
}
