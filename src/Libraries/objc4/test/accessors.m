/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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

// TEST_CONFIG OS=!exclavekit MEM=mrc,arc
// TEST_CFLAGS -framework Foundation

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/objc-abi.h>
#include "test.h"

@interface Test : NSObject {
@public
    NSString *_value;
    // _object is at the last optimized property offset
    id _object __attribute__((aligned(64)));
}
@property(readonly) Class cls;
@property(copy) NSString *value;
@property(assign) id object;
@end

typedef struct {
    void *isa;
    void *_value;
    // _object is at the last optimized property offset
    void *_object __attribute__((aligned(64)));
} TestDefs;

@implementation Test

// Question:  why can't this code be automatically generated?

#if !__has_feature(objc_arc)
- (void)dealloc {
    self.value = nil;
    self.object = nil;
    [super dealloc];
}
#endif

- (Class)cls { return objc_getProperty(self, _cmd, 0, YES); }

- (NSString*)value { return (NSString*) objc_getProperty(self, _cmd, offsetof(TestDefs, _value), YES); }
- (void)setValue:(NSString*)inValue { objc_setProperty(self, _cmd, offsetof(TestDefs, _value), inValue, YES, YES); }

- (id)object { return objc_getProperty(self, _cmd, offsetof(TestDefs, _object), YES); }
- (void)setObject:(id)inObject { objc_setProperty(self, _cmd, offsetof(TestDefs, _object), inObject, YES, NO); }

- (NSString *)description {
    return [NSString stringWithFormat:@"value = %@, object = %@", self.value, self.object];
}

@end

@interface TestUninitializedClass: NSObject @end
@implementation TestUninitializedClass @end

int main() {
    PUSH_POOL {

        NSMutableString *value = [NSMutableString stringWithUTF8String:"test"];
        id object = [NSNumber numberWithInt:11];
        Test *t = AUTORELEASE([Test new]);
        t.value = value;
        [value setString:@"yuck"];      // mutate the string.
        testassert(t.value != value);   // must copy, since it was mutable.
        testassert([t.value isEqualToString:@"test"]);

        Class testClass = [Test class];
        Class cls = t.cls;
        testassert(testClass == cls);
        cls = t.cls;
        testassert(testClass == cls);

        t.object = object;
        t.object = object;

        // Make sure we handle getting a property set to a class that hasn't yet
        // run +initialize. rdar://113033917
        t.object = nil;
        t->_object = objc_getClass("TestUninitializedClass");
        testassert(t.object == objc_getClass("TestUninitializedClass"));

        // NSLog(@"t.object = %@, t.value = %@", t.object, t.value);
        // NSLog(@"t.object = %@, t.value = %@", t.object, t.value); // second call will optimized getters.

    } POP_POOL;

    succeed(__FILE__);

    return 0;
}
