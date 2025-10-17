/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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


#ifdef __OBJC__

#import <Foundation/Foundation.h>
#import <Security/SecEscrowRequest.h>

NS_ASSUME_NONNULL_BEGIN

NSXPCInterface* SecEscrowRequestSetupControlProtocol(NSXPCInterface* interface);


@protocol EscrowRequestXPCProtocol <NSObject>

- (void)triggerEscrowUpdate:(NSString*)reason
                    options:(NSDictionary*)options
                      reply:(void (^)(NSError* _Nullable error))reply;

- (void)cachePrerecord:(NSString*)uuid
   serializedPrerecord:(nonnull NSData *)prerecord
                 reply:(nonnull void (^)(NSError * _Nullable))reply;

- (void)fetchPrerecord:(NSString*)prerecordUUID
                 reply:(void (^)(NSData* _Nullable serializedPrerecord, NSError* _Nullable error))reply;

- (void)fetchRequestWaitingOnPasscode:(void (^)(NSString* _Nullable requestUUID, NSError* _Nullable error))reply;

- (void)fetchRequestStatuses:(void (^)(NSDictionary<NSString*, NSString*>* _Nullable requestUUID, NSError* _Nullable error))reply;

- (void)resetAllRequests:(void (^)(NSError* _Nullable error))reply;

- (void)storePrerecordsInEscrow:(void (^)(uint64_t count, NSError* _Nullable error))reply;

- (void)escrowCompletedWithinLastSeconds:(NSTimeInterval)timeInterval
                                   reply:(void (^)(BOOL escrowCompletedWithin, NSError* _Nullable error))reply;

@end

NS_ASSUME_NONNULL_END

#endif /* __OBJC__ */
