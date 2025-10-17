/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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
//  SOSPeerRateLimiter.h
//  Security
//
#import "keychain/ckks/RateLimiter.h"
#include "keychain/SecureObjectSync/SOSPeer.h"

#ifndef SOSPeerRateLimiter_h
#define SOSPeerRateLimiter_h

enum RateLimitState{
    RateLimitStateCanSend = 1,
    RateLimitStateHoldMessage = 2
};

@interface  PeerRateLimiter : RateLimiter
{
    NSString *peerID;
}

@property (retain) NSString *peerID;
@property (retain) NSMutableDictionary *accessGroupRateLimitState;
@property (retain) NSMutableDictionary *accessGroupToTimer;
@property (retain) NSMutableDictionary *accessGroupToNextMessageToSend;

-(instancetype)initWithPeer:(SOSPeerRef)peer;
-(NSDictionary *) setUpConfigForPeer;
-(enum RateLimitState) stateForAccessGroup:(NSString*) accessGroup;
@end

@interface KeychainItem : NSObject
@property (atomic, retain) NSString* accessGroup;
-(instancetype) initWithAccessGroup:(NSString*)accessGroup;
@end

#endif /* SOSPeerRateLimiter_h */
