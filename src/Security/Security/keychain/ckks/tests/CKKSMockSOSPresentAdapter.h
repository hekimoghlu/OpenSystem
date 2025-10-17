/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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

#import "keychain/ot/OTSOSAdapter.h"
#include "keychain/SecureObjectSync/SOSAccount.h"

NS_ASSUME_NONNULL_BEGIN

@interface CKKSMockSOSPresentAdapter : NSObject <OTSOSAdapter>

// If you fill these in, the OTSOSAdapter methods will error with these errors.
@property (nullable) NSError* selfPeerError;
@property (nullable) NSError* trustedPeersError;

@property BOOL aksLocked;

@property bool excludeSelfPeerFromTrustSet;

@property SOSCCStatus circleStatus;
@property (nullable) NSError* circleStatusError;

@property CKKSSOSSelfPeer* selfPeer;
@property NSMutableSet<id<CKKSSOSPeerProtocol>>* trustedPeers;

@property BOOL safariViewEnabled;

@property BOOL ckks4AllStatus;
@property BOOL ckks4AllStatusIsSet;

@property bool joinAfterRestoreCircleStatusOverride;
@property bool joinAfterRestoreResult;

@property bool resetToOfferingCircleStatusOverride;
@property bool resetToOfferingResult;

@property (nullable) void (^updateOctagonKeySetListener)(id<CKKSSelfPeer>);

- (instancetype)initWithSelfPeer:(CKKSSOSSelfPeer*)selfPeer
                    trustedPeers:(NSSet<id<CKKSSOSPeerProtocol>>*)trustedPeers
                       essential:(BOOL)essential;

- (NSSet<id<CKKSRemotePeerProtocol>>*)allPeers;

- (void)setSOSEnabled:(bool)isEnabled;

@end

NS_ASSUME_NONNULL_END

#endif // OCTAGON
