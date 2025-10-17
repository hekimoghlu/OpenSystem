/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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

//
//  SOSAccountTransaction.h
//  sec
//
//
//

#ifndef SOSAccountTransaction_h
#define SOSAccountTransaction_h

#include <CoreFoundation/CoreFoundation.h>
#include "keychain/SecureObjectSync/SOSAccountPriv.h"
#include <CoreFoundation/CFRuntime.h>

NS_ASSUME_NONNULL_BEGIN

@class SOSAccountTransaction;

@interface SOSAccount (Transaction)

+ (void)performOnQuietAccountQueue:(void (^)(void))action;

- (void) performTransaction: (void (^)(SOSAccountTransaction* txn)) action;
- (void) performTransaction_Locked: (void (^)(SOSAccountTransaction* txn)) action;

@end

@interface SOSAccountTransaction : NSObject

- (instancetype) init NS_UNAVAILABLE;
- (instancetype) initWithAccount: (SOSAccount*) account quiet:(bool)quiet NS_DESIGNATED_INITIALIZER;

- (void) finish;
- (void) restart;

- (void) requestSyncWith: (NSString*) peerID;
- (void) requestSyncWithPeers: (NSSet<NSString*>*) peerList;

@property SOSAccount *account;

@property (readonly) NSString* description;

@end

NS_ASSUME_NONNULL_END

#endif /* SOSAccountTransaction_h */
