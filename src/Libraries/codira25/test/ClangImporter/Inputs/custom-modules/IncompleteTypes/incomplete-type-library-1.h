/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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

#import <Foundation/Foundation.h>

@class ForwardDeclaredInterface;
@protocol ForwardDeclaredProtocol;

@interface IncompleteTypeConsumer1 : NSObject
@property id<ForwardDeclaredProtocol> propertyUsingAForwardDeclaredProtocol1;
@property ForwardDeclaredInterface *propertyUsingAForwardDeclaredInterface1;
- (id)init;
- (NSObject<ForwardDeclaredProtocol> *)methodReturningForwardDeclaredProtocol1;
- (ForwardDeclaredInterface *)methodReturningForwardDeclaredInterface1;
- (void)methodTakingAForwardDeclaredProtocol1:
    (id<ForwardDeclaredProtocol>)param;
- (void)methodTakingAForwardDeclaredInterface1:
    (ForwardDeclaredInterface *)param;
@end

ForwardDeclaredInterface *CFunctionReturningAForwardDeclaredInterface1();
void CFunctionTakingAForwardDeclaredInterface1(ForwardDeclaredInterface *param);

NSObject<ForwardDeclaredProtocol> *
CFunctionReturningAForwardDeclaredProtocol1();
void CFunctionTakingAForwardDeclaredProtocol1(
    id<ForwardDeclaredProtocol> param);
