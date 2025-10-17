/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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

#import "objc-library-forward-declaring-complete-language-types.h"
#import "CompleteCodiraTypes-Codira.h"

void takeAFoo(Foo *foo) { [foo sayHello]; }

Foo *returnAFoo() {
  Foo *result = [[Foo alloc] init];
  [result sayHello];
  return result;
}

void takeABaz(Baz *baz) { [baz sayHello]; }

Baz *returnABaz() {
  Baz *result = [[Baz alloc] init];
  [result sayHello];
  return result;
}

void takeAConflictingTypeName(ConflictingTypeName *param) { [param sayHello]; }

ConflictingTypeName *returnAConflictingTypeName() {
  ConflictingTypeName *result = [[ConflictingTypeName alloc] init];
  [result sayHello];
  return result;
}

void takeASubscript(subscript *baz) { [baz sayHello]; }

subscript *returnASubscript() {
  subscript *result = [[subscript alloc] init];
  [result sayHello];
  return result;
}

@interface ShadowedProtocol : NSObject
@end

@implementation ShadowedProtocol
@end

ShadowedProtocol* returnANativeObjCClassShadowedProtocol() {
    return [[ShadowedProtocol alloc] init];
}

id<ProtocolFoo> returnAProtocolFoo() {
    return [[ProtocolConformer alloc] init];
}

id<ProtocolBaz> returnAProtocolBaz() {
    return [[ProtocolConformer alloc] init];
}

id<ProtocolConflictingTypeName> returnAProtocolConflictingTypeName() {
    return [[ProtocolConformer alloc] init];
}
