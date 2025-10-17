/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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

#import "ObjCClasses.h"
#import <Foundation/NSError.h>
#include <stdio.h>
#include <assert.h>

@implementation HasHiddenIvars
@synthesize x;
@synthesize y;
@synthesize z;
@synthesize t;
@end

@implementation HasHiddenIvars2
@synthesize x;
@synthesize y;
@synthesize z;
@end

@implementation TestingNSError
+ (BOOL)throwNilError:(NSError **)error {
  return 0;
}

+ (nullable void *)maybeThrow:(BOOL)shouldThrow error:(NSError **)error {
  if (shouldThrow) {
    *error = [NSError errorWithDomain:@"pointer error" code:0 userInfo:nil];
    return 0;
  }
  return (void *)42;
}

+ (nullable void (^)(void))blockThrowError:(NSError **)error {
  *error = [NSError errorWithDomain:@"block error" code:0 userInfo:nil];
  return 0;
}

@end

@implementation Container
- (id)initWithObject:(id)anObject {
  if ((self = [super init]) != nil) {
    self.object = anObject;
  }
  return self;
}

- (void)processObjectWithBlock:(void (^)(id))block {
  block(self.object);
}

- (void)updateObjectWithBlock:(id (^)())block {
  self.object = block();
}

@synthesize object;

- (id)initWithCat1:(id)anObject {
  return [self initWithObject:anObject];
}

- (id)getCat1 {
  return self.object;
}

- (void)setCat1:(id)obj {
  self.object = obj;
}

- (id)cat1Property {
  return self.object;
}

- (void)setCat1Property:(id)prop {
  self.object = prop;
}

@end

@implementation SubContainer
@end

@implementation NestedContainer
@end

@implementation StringContainer
@end

@implementation CopyingContainer
@end

@implementation Animal
- (NSString *)noise {
  return @"eep";
}
@end

@implementation Dog
- (NSString *)noise {
  return @"woof";
}
@end

@implementation AnimalContainer
@end

#if __has_feature(objc_class_property)
static int _value = 0;
@implementation ClassWithClassProperty
+ (int)value {
  return _value;
}
+ (void)setValue:(int)newValue {
  _value = newValue;
}
+ (void)reset {
  _value = 0;
}
@end

@implementation ObjCSubclassWithClassProperty
+ (BOOL)optionalClassProp {
  return YES;
}
@end

@implementation PropertyNamingConflict
- (id)prop { return self; }
+ (id)prop { return nil; }
@end

#endif

@implementation BridgedInitializer
- (id) initWithArray: (NSArray*) array {
  _objects = array;
  return self;
}
- (NSInteger) count {
  return _objects.count;
}
@end

static unsigned counter = 0;

@implementation NSLifetimeTracked

+ (id) allocWithZone:(NSZone *)zone {
  counter++;
  return [super allocWithZone:zone];
}

- (void) dealloc {
  counter--;
}

+ (unsigned) count {
  return counter;
}

@end

@implementation TestingBool

- (void) shouldBeTrueObjCBool: (BOOL)value {
  assert(value);
}

- (void) shouldBeTrueCBool: (_Bool)value {
  assert(value);
}

@end

@implementation OuterType

- (id)init {
  if ((self = [super init]) != nil) {
  }
  return self;
}

@end

@implementation OuterTypeInnerType

- (id)init {
  if ((self = [super init]) != nil) {
    self.things = [NSArray array];
  }
  return self;
}

@end

@implementation ObjCPrintOnDealloc
- (void)dealloc {
  printf("ObjCPrintOnDealloc deinitialized!\n");
}
@end
