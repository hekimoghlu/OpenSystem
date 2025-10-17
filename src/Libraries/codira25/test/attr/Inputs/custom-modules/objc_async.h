/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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

@import Foundation;

#pragma clang assume_nonnull begin

typedef void (^CompletionHandler)(NSInteger);

@interface HandlerTest : NSObject
-(void)simpleWithCompletionHandler:(void (^)(NSInteger))handler;
-(void)simpleArg:(NSInteger)arg completionHandler:(void (^)(NSInteger))handler;
-(void)aliasWithCompletionHandler:(CompletionHandler)handler;
-(void)errorWithCompletionHandler:(void (^)(NSString *_Nullable, NSError * _Nullable))handler;
-(BOOL)removedError:(NSError * _Nullable * _Nullable)error completionHandler:(void (^)(NSString *_Nullable, NSError * _Nullable))handler;

-(void)asyncImportSame:(NSInteger)arg completionHandler:(void (^)(NSInteger))handler;
-(void)asyncImportSame:(NSInteger)arg replyTo:(void (^)(NSInteger))handler __attribute__((language_async(none)));

-(void)simpleOnMainActorWithCompletionHandler:(void (^)(NSInteger))handler __attribute__((language_attr("@MainActor")));

@end

#pragma clang assume_nonnull end
