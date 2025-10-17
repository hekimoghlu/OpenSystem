/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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


#if OCTAGON

#import <Foundation/Foundation.h>
#import "keychain/ckks/CKKSPeer.h"
#import "keychain/ckks/CKKSPeerProvider.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperSpecificUser.h"
#import "keychain/ot/CuttlefishXPCWrapper.h"
#import "keychain/ot/OTPersonaAdapter.h"

NS_ASSUME_NONNULL_BEGIN

@interface OctagonSelfPeer : NSObject <CKKSSelfPeer>

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithPeerID:(NSString*)peerID
               signingIdentity:(SFIdentity*)signingIdentity
            encryptionIdentity:(SFIdentity*)encryptionIdentity;

@end

@interface OctagonCKKSPeerAdapter : NSObject  <CKKSPeerProvider>

@property (nullable) NSString* peerID;
@property (readonly) TPSpecificUser* specificUser;
@property (readonly) CuttlefishXPCWrapper* cuttlefishXPCWrapper;
@property id<OTPersonaAdapter> personaAdapter;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithPeerID:(NSString*)peerID
                  specificUser:(TPSpecificUser*)specificUser
                personaAdapter:(id<OTPersonaAdapter>)personaAdapter
                 cuttlefishXPC:(CuttlefishXPCWrapper*)cuttlefishXPCWrapper;
@end

NS_ASSUME_NONNULL_END

#endif
