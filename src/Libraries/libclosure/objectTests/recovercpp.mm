/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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
// TEST_CONFIG rdar://6214670
// TEST_CFLAGS -framework Foundation

#import <Foundation/Foundation.h>
#import <Block.h>
#import "test.h"

int constructors = 0;
int destructors = 0;


#import <Block_private.h>

#define CONST const

class TestObject
{
public:
	TestObject(CONST TestObject& inObj);
	TestObject();
	~TestObject();
	
	TestObject& operator=(CONST TestObject& inObj);

	int version() CONST { return _version; }
private:
	mutable int _version;
};

TestObject::TestObject(CONST TestObject& inObj)
	
{
        ++constructors;
        _version = inObj._version;
	//printf("%p (%d) -- TestObject(const TestObject&) called\n", this, _version); 
}


TestObject::TestObject()
{
        _version = ++constructors;
	//printf("%p (%d) -- TestObject() called\n", this, _version); 
}


TestObject::~TestObject()
{
	//printf("%p -- ~TestObject() called\n", this);
        ++destructors;
}


TestObject& TestObject::operator=(CONST TestObject& inObj)
{
	//printf("%p -- operator= called\n", this);
        _version = inObj._version;
	return *this;
}


void testRoutine() {
    TestObject one;
    
    void (^b)(void) = [^{ printf("my copy of one is %d\n", one.version()); } copy];
#if 0
// just try one copy, one release
    for (int i = 0; i < 10; ++i)
        [b retain];
    for (int i = 0; i < 10; ++i)
        [b release];
    for (int i = 0; i < 10; ++i)
        Block_copy(b);
    for (int i = 0; i < 10; ++i)
        Block_release(b);
#endif
    [b release];
}

int main() {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    //testGC();
    for (int i = 0; i < 200; ++i)   // do enough to trigger TLC if GC is on
        testRoutine();
    [pool drain];

    if (constructors != destructors) {
        fail("didn't recover %d const copies", constructors - destructors);
    }

    succeed(__FILE__);
}
