/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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

#include <Foundation.h>

@class ForwardDeclaredInterface;
@protocol ForwardDeclaredProtocol;

@interface Bar : NSObject
@property id<ForwardDeclaredProtocol> propertyUsingAForwardDeclaredProtocol;
@property ForwardDeclaredInterface* propertyUsingAForwardDeclaredInterface;
- (NSObject<ForwardDeclaredProtocol> *) methodReturningForwardDeclaredProtocol;
- (ForwardDeclaredInterface *) methodReturningForwardDeclaredInterface;
- (int)methodTakingAForwardDeclaredProtocolAsAParameter:(id<ForwardDeclaredProtocol>)param1;
- (int)methodTakingAForwardDeclaredInterfaceAsAParameter:(ForwardDeclaredInterface *)param1 andAnother:(ForwardDeclaredInterface *)param2;
@end

ForwardDeclaredInterface* CFunctionReturningAForwardDeclaredInterface();
void CFunctionTakingAForwardDeclaredInterfaceAsAParameter(ForwardDeclaredInterface* param1);

NSObject<ForwardDeclaredProtocol> *CFunctionReturningAForwardDeclaredProtocol();
void CFunctionTakingAForwardDeclaredProtocolAsAParameter(id<ForwardDeclaredProtocol> param1);

@interface CompleteInterface
@end
@protocol CompleteProtocol
@end

@interface Foo : NSObject
@property id<CompleteProtocol> propertyUsingACompleteProtocol;
@property CompleteInterface *propertyUsingACompleteInterface;
- (NSObject<CompleteProtocol> *)methodReturningCompleteProtocol;
- (CompleteInterface *)methodReturningCompleteInterface;
- (int)methodTakingACompleteProtocolAsAParameter:(id<CompleteProtocol>)param1;
- (int)methodTakingACompleteInterfaceAsAParameter:(CompleteInterface *)param1
                                       andAnother:(CompleteInterface *)param2;
@end

CompleteInterface *CFunctionReturningACompleteInterface();
void CFunctionTakingACompleteInterfaceAsAParameter(CompleteInterface *param1);

NSObject<CompleteProtocol> *CFunctionReturningACompleteProtocol();
void CFunctionTakingACompleteProtocolAsAParameter(id<CompleteProtocol> param1);
