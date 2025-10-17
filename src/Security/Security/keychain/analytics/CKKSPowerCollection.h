/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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

#if OCTAGON


@protocol CKKSPowerEventType <NSObject>
@end
typedef NSString<CKKSPowerEventType> CKKSPowerEvent;

extern CKKSPowerEvent* const kCKKSPowerEventOutgoingQueue;
extern CKKSPowerEvent* const kCKKSPowerEventIncommingQueue;
extern CKKSPowerEvent* const kCKKSPowerEventTLKShareProcessing;
extern CKKSPowerEvent* const kCKKSPowerEventScanLocalItems;
extern CKKSPowerEvent* const kCKKSPowerEventFetchAllChanges;
extern CKKSPowerEvent* const kCKKSPowerEventReencryptOutgoing;

@protocol OTPowerEventType <NSObject>
@end
typedef NSString<OTPowerEventType> OTPowerEvent;

extern OTPowerEvent* const kOTPowerEventRestore;
extern OTPowerEvent* const kOTPowerEventEnroll;

@class CKKSOutgoingQueueEntry;

@interface CKKSPowerCollection : NSOperation

+ (void)CKKSPowerEvent:(CKKSPowerEvent *)operation zone:(NSString *)zone;
+ (void)CKKSPowerEvent:(CKKSPowerEvent *)operation zone:(NSString *)zone count:(NSUInteger)count;
+ (void)CKKSPowerEvent:(CKKSPowerEvent *)operation count:(NSUInteger)count;

+ (void)OTPowerEvent:(NSString *)operation;

- (void)storedOQE:(CKKSOutgoingQueueEntry *)oqe;
- (void)deletedOQE:(CKKSOutgoingQueueEntry *)oqe;

- (void)commit;

@end

#endif
