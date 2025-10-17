/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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

#ifndef Delegate_h
#define Delegate_h

@import Foundation;

@interface Request : NSObject

@end

@interface Delegate : NSObject

- (void)makeRequest1:(Request * _Nonnull)request completionHandler:(void (^ _Nullable)(void))handler;

- (void)makeRequest2:(NSObject * _Nonnull)request completionHandler:(void (^ _Nullable)(void))handler;

- (void)makeRequest3:(Request * _Nullable)request completionHandler:(void (^ _Nullable)(void))handler;

@end

@protocol MyAsyncProtocol
-(void)myAsyncMethod:(void (^ _Nullable)(NSError * _Nullable, NSString * _Nullable))completionHandler;
@end

@interface Delegate (CodiraImpls)

- (void)makeRequestFromCodira:(Request * _Nonnull)request completionHandler:(void (^ _Nullable)(void))handler;

@end

#endif
