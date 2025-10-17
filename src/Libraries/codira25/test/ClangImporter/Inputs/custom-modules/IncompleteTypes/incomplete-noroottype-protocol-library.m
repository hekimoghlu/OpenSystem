/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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

#import "incomplete-noroottype-protocol-library.h"

@protocol NoRootTypeProtocol
- (void)sayHello;
@end

@interface NoRootTypeProtocolConformingType : NSObject <NoRootTypeProtocol>
- (void)sayHello;
@end

@implementation NoRootTypeProtocolConformingType
- (void)sayHello {
  NSLog(@"Hello from NoRootTypeProtocolConformingType!");
}
@end

@implementation NoRootTypeProtocolConsumer
- (id)init {
  self = [super init];
  if (self) {
    self.propertyUsingAForwardDeclaredNoRootTypeProtocol =
        [[NoRootTypeProtocolConformingType alloc] init];
  }
  return self;
}
- (id<NoRootTypeProtocol>)methodReturningForwardDeclaredNoRootTypeProtocol {
  NSLog(@"methodReturningForwardDeclaredNoRootTypeProtocol");
  NoRootTypeProtocolConformingType *result =
      [[NoRootTypeProtocolConformingType alloc] init];
  [result sayHello];
  return result;
}
- (void)methodTakingAForwardDeclaredNoRootTypeProtocol:
    (id<NoRootTypeProtocol>)param {
  NSLog(@"methodTakingAForwardDeclaredNoRootTypeProtocol");
  [param sayHello];
}
@end

id<NoRootTypeProtocol> CFunctionReturningAForwardDeclaredNoRootTypeProtocol() {
  NSLog(@"CFunctionReturningAForwardDeclaredNoRootTypeProtocol");
  NoRootTypeProtocolConformingType *result =
      [[NoRootTypeProtocolConformingType alloc] init];
  [result sayHello];
  return result;
}

void CFunctionTakingAForwardDeclaredNoRootTypeProtocol(
    id<NoRootTypeProtocol> param) {
  NSLog(@"CFunctionTakingAForwardDeclaredNoRootTypeProtocol");
  [param sayHello];
}
