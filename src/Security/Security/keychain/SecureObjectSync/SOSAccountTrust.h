/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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
//  SOSAccountTrust_h
//  Security

#ifndef SOSAccountTrust_h
#define SOSAccountTrust_h

#import <Foundation/Foundation.h>
#import "keychain/SecureObjectSync/SOSCircle.h"
#import "keychain/SecureObjectSync/SOSFullPeerInfo.h"
#import "keychain/SecureObjectSync/SOSRing.h"

typedef bool (^SOSModifyCircleBlock)(SOSCircleRef circle);
typedef void (^SOSIteratePeerBlock)(SOSPeerInfoRef peerInfo);
typedef bool (^SOSModifyPeerBlock)(SOSPeerInfoRef peerInfo);
typedef bool (^SOSModifyPeerInfoBlock)(SOSFullPeerInfoRef fpi, CFErrorRef *error);
typedef SOSRingRef(^RingNameBlock)(CFStringRef name, SOSRingRef ring);
typedef void (^SOSModifyPeersInCircleBlock)(SOSCircleRef circle, CFMutableArrayRef appendPeersTo);

@interface SOSAccountTrust : NSObject
{
   NSMutableDictionary *   expansion;

    SOSFullPeerInfoRef      fullPeerInfo;
    SOSPeerInfoRef          peerInfo;
    NSString*               peerID;

    SOSCircleRef            trustedCircle;
    NSMutableSet *          retirees;
    enum DepartureReason    departureCode;

    SecKeyRef               _cachedOctagonSigningKey;
    SecKeyRef               _cachedOctagonEncryptionKey;
}
@property (strong, nonatomic)   NSMutableDictionary *   expansion;

@property (nonatomic)           SOSFullPeerInfoRef      fullPeerInfo;

// Convenince getters
@property (nonatomic, readonly) SOSPeerInfoRef          peerInfo;
@property (nonatomic, readonly) NSString*               peerID;


@property (nonatomic)           SOSCircleRef            trustedCircle;
@property (strong, nonatomic)   NSMutableSet *          retirees;
@property (nonatomic)           enum DepartureReason    departureCode;

@property (assign)              SecKeyRef               cachedOctagonSigningKey;
@property (assign)              SecKeyRef               cachedOctagonEncryptionKey;

+(instancetype)trust;

-(id)init;
-(id)initWithRetirees:(NSMutableSet*)retirees fpi:(SOSFullPeerInfoRef)identity circle:(SOSCircleRef) trusted_circle
        departureCode:(enum DepartureReason)code peerExpansion:(NSMutableDictionary*)expansion;


@end

#endif /* Trust_h */
