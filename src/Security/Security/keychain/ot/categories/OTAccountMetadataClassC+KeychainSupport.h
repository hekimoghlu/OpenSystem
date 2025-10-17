/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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

#import "keychain/ot/proto/generated_source/OTAccountMetadataClassC.h"
#import "keychain/ot/OTPersonaAdapter.h"

NS_ASSUME_NONNULL_BEGIN

@interface OTAccountMetadataClassC (KeychainSupport)

- (BOOL)saveToKeychainForContainer:(NSString*)containerName
                         contextID:(NSString*)contextID
                   personaAdapter:(id<OTPersonaAdapter>)personaAdapter
               personaUniqueString:(NSString* _Nullable)personaUniqueString
                             error:(NSError**)error;

+ (BOOL)deleteFromKeychainForContainer:(NSString*)containerName
                             contextID:(NSString*)contextID
                       personaAdapter:(id<OTPersonaAdapter>)personaAdapter
                   personaUniqueString:(NSString* _Nullable)personaUniqueString
                                 error:(NSError**)error  __attribute__((swift_error(nonnull_error)));

+ (OTAccountMetadataClassC* _Nullable)loadFromKeychainForContainer:(NSString*)containerName
                                                         contextID:(NSString*)contextID
                                                   personaAdapter:(id<OTPersonaAdapter>)personaAdapter
                                               personaUniqueString:(NSString* _Nullable)personaUniqueString
                                                             error:(NSError**)error;
@end

@class TPSyncingPolicy;
@class CKKSTLKShare;
@class OTSecureElementPeerIdentity;
@class TPPBSecureElementIdentity;
@interface OTAccountMetadataClassC (NSSecureCodingSupport)
- (void)setTPSyncingPolicy:(TPSyncingPolicy* _Nullable)policy;
- (TPSyncingPolicy* _Nullable)getTPSyncingPolicy;

- (void)setTLKSharesPairedWithVoucher:(NSArray<CKKSTLKShare*>*)newTLKShares;
- (NSArray<CKKSTLKShare*>*)getTLKSharesPairedWithVoucher;

- (void)setOctagonSecureElementIdentity:(OTSecureElementPeerIdentity *)secureElementIdentity;
- (TPPBSecureElementIdentity* _Nullable)parsedSecureElementIdentity;
@end

NS_ASSUME_NONNULL_END

#endif // OCTAGON
