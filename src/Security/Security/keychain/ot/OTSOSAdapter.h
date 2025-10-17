/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 18, 2023.
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

#import "keychain/ckks/CKKSPeer.h"
#import "keychain/ckks/CKKSPeerProvider.h"
#import "keychain/ot/OTDefines.h"
#import "keychain/SecureObjectSync/SOSCloudCircle.h"

NS_ASSUME_NONNULL_BEGIN

@protocol OTSOSAdapter <CKKSPeerProvider>
@property (nonatomic, readonly) bool sosEnabled;
- (SOSCCStatus)circleStatus:(NSError**)error;
- (id<CKKSSelfPeer> _Nullable)currentSOSSelf:(NSError**)error;
- (NSSet<id<CKKSRemotePeerProtocol>>* _Nullable)fetchTrustedPeers:(NSError**)error;
- (BOOL)updateOctagonKeySetWithAccount:(id<CKKSSelfPeer>)currentSelfPeer error:(NSError**)error;
- (BOOL)preloadOctagonKeySetOnAccount:(id<CKKSSelfPeer>)currentSelfPeer error:(NSError**)error;
- (BOOL)updateCKKS4AllStatus:(BOOL)status error:(NSError**)error;

- (BOOL)safariViewSyncingEnabled:(NSError**)error __attribute__((swift_error(nonnull_error)));

- (bool)joinAfterRestore:(NSError * _Nullable __autoreleasing * _Nullable)error;
- (bool)resetToOffering:(NSError * _Nullable __autoreleasing * _Nullable)error;
@end

@interface OTSOSActualAdapter : NSObject <OTSOSAdapter>
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initAsEssential:(BOOL)essential;

// Helper methods.
+ (NSSet<NSString*>*)sosCKKSViewList;
@end


// This adapter is for a platform which does not have SOS (e.g., aTV, Watch, HomePod)
@interface OTSOSMissingAdapter : NSObject <OTSOSAdapter>
@end

// Helper code
@interface OTSOSAdapterHelpers : NSObject
+ (NSArray<NSData*>* _Nullable)peerPublicSigningKeySPKIsForCircle:(id<OTSOSAdapter>)sosAdapter error:(NSError**)error;
@end

NS_ASSUME_NONNULL_END

#endif // OCTAGON
