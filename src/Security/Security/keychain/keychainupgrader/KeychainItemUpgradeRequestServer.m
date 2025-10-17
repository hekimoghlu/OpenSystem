/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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


#import "utilities/debugging.h"

#import "keychain/keychainupgrader/KeychainItemUpgradeRequestController.h"
#import "keychain/keychainupgrader/KeychainItemUpgradeRequestServer.h"

#import "keychain/ckks/CKKSLockStateTracker.h"
#import "keychain/ckks/CKKSAnalytics.h"

@implementation KeychainItemUpgradeRequestServer

- (instancetype)initWithLockStateTracker:(CKKSLockStateTracker*)lockStateTracker
{
    if((self = [super init])) {
        _controller = [[KeychainItemUpgradeRequestController alloc] initWithLockStateTracker:lockStateTracker];
    }
    return self;
}

+ (KeychainItemUpgradeRequestServer*)server
{
    static KeychainItemUpgradeRequestServer* server;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        server = [[KeychainItemUpgradeRequestServer alloc] initWithLockStateTracker:[CKKSLockStateTracker globalTracker]];
    });
    return server;
}

@end
