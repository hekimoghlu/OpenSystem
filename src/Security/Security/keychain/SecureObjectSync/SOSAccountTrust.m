/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
//  SOSAccountTrust.c
//  Security
//
#import "keychain/SecureObjectSync/SOSAccountPriv.h"
#import "keychain/SecureObjectSync/SOSAccountTrust.h"

@implementation SOSAccountTrust
@synthesize cachedOctagonEncryptionKey = _cachedOctagonEncryptionKey;
@synthesize cachedOctagonSigningKey = _cachedOctagonSigningKey;

+(instancetype)trust
{
    return [[SOSAccountTrust alloc]init];
}

-(id)init
{
    if ((self = [super init])) {
        self.retirees = [NSMutableSet set];
        self.fullPeerInfo = NULL;
        self.trustedCircle = NULL;
        self.departureCode = kSOSDepartureReasonError;
        self.expansion = [NSMutableDictionary dictionary];
    }
    return self;
}

-(id)initWithRetirees:(NSMutableSet*)r fpi:(SOSFullPeerInfoRef)fpi circle:(SOSCircleRef) trusted_circle
        departureCode:(enum DepartureReason)code peerExpansion:(NSMutableDictionary*)e
{

    if ((self = [super init])) {
        self.retirees = r;
        self.fullPeerInfo = fpi;
        self.trustedCircle = trusted_circle;
        self.departureCode = code;
        self.expansion = e;
    }
    return self;
}
- (void)dealloc {
    if(self) {
        CFReleaseNull(self->fullPeerInfo);
        CFReleaseNull(self->peerInfo);
        CFReleaseNull(self->trustedCircle);
        CFReleaseNull(self->_cachedOctagonSigningKey);
        CFReleaseNull(self->_cachedOctagonEncryptionKey);
    }
}

- (SOSPeerInfoRef) peerInfo {
    return SOSFullPeerInfoGetPeerInfo(self.fullPeerInfo);
}

- (NSString*) peerID {
    return (__bridge_transfer NSString*) CFRetainSafe(SOSPeerInfoGetPeerID(self.peerInfo));
}

@synthesize trustedCircle = trustedCircle;

- (void) setTrustedCircle:(SOSCircleRef) circle {
    CFRetainAssign(self->trustedCircle, circle);
}

@synthesize retirees = retirees;

-(void) setRetirees:(NSSet *)newRetirees
{
    self->retirees = newRetirees.mutableCopy;
}

@synthesize fullPeerInfo = fullPeerInfo;

- (void) setFullPeerInfo:(SOSFullPeerInfoRef) newIdentity {
    CFRetainAssign(self->fullPeerInfo, newIdentity);
}

@synthesize expansion = expansion;

-(void)setExpansion:(NSDictionary*) newExpansion
{
    self->expansion = newExpansion.mutableCopy;
}

@synthesize departureCode = departureCode;

-(void)setDepartureCode:(enum DepartureReason)newDepartureCode
{
    self->departureCode = newDepartureCode;
}

@end

