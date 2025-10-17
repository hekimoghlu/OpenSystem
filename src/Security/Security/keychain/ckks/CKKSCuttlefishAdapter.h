/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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

#import <SecurityFoundation/SecurityFoundation.h>

#import "keychain/ckks/CKKSCuttlefishAdapter.h"
#import "keychain/ckks/CKKSTLKShareRecord.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperSpecificUser.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "keychain/ot/CuttlefishXPCWrapper.h"

NS_ASSUME_NONNULL_BEGIN

@protocol CKKSCuttlefishAdapterProtocol

- (void)fetchCurrentItem:(TPSpecificUser* __nullable)activeAccount
                   items:(nonnull NSArray<CuttlefishCurrentItemSpecifier *> *)items
                   reply:(nonnull void (^)(NSArray<CuttlefishCurrentItem *> * _Nullable, NSArray<CKRecord *> * _Nullable, NSError * _Nullable))reply;

- (void)fetchPCSIdentityByKey:(TPSpecificUser* __nullable)activeAccount
                  pcsservices:(nonnull NSArray<CuttlefishPCSServiceIdentifier *> *)pcsservices
                        reply:(nonnull void (^)(NSArray<CuttlefishPCSIdentity *> * _Nullable, NSArray<CKRecord *> * _Nullable, NSError * _Nullable))reply;

- (void)fetchRecoverableTLKShares:(TPSpecificUser* __nullable)activeAccount
                           peerID:(NSString*)peerID
                        contextID:(NSString*)contextID
                          altDSID:(NSString* _Nullable)altDSID
                           flowID:(NSString* _Nullable)flowID
                  deviceSessionID:(NSString* _Nullable)deviceSessionID
                   canSendMetrics:(BOOL)canSendMetrics
                            reply:(void (^)(NSArray<CKKSTLKShareRecord*>* _Nullable tlkShares,
                                            NSError * _Nullable error))reply;

@end


@interface CKKSCuttlefishAdapter : NSObject<CKKSCuttlefishAdapterProtocol>

@property CuttlefishXPCWrapper* cuttlefishXPCWrapper;

- (instancetype)initWithConnection:(id<NSXPCProxyCreating>)cuttlefishXPCConnection;

NS_ASSUME_NONNULL_END

@end

#else   // !OCTAGON
#import <Foundation/Foundation.h>
@interface CKKSCuttlefishAdapter : NSObject
{
    
}
@end
#endif  // OCTAGON
