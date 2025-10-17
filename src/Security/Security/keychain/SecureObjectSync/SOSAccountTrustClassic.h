/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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
//  SOSAccountTrustClassic.h
//  Security
//

#ifndef SOSAccountTrustClassic_h
#define SOSAccountTrustClassic_h

#import "keychain/SecureObjectSync/SOSAccountTransaction.h"
#import "keychain/SecureObjectSync/SOSAccountTrust.h"
#import "keychain/SecureObjectSync/SOSTypes.h"
#import "keychain/SecureObjectSync/SOSTransportCircleKVS.h"
@class SOSAccount;

@interface SOSAccountTrustClassic : SOSAccountTrust

+(instancetype)trustClassic;

//Gestalt Dictionary
-(bool) updateGestalt:(SOSAccount*)account newGestalt:(CFDictionaryRef) new_gestalt;

//Peers
-(SOSPeerInfoRef) copyPeerWithID:(CFStringRef) peerid err:(CFErrorRef *)error;
-(bool) isAccountIdentity:(SOSPeerInfoRef)peerInfo err:(CFErrorRef *)error;
-(SecKeyRef) copyPublicKeyForPeer:(CFStringRef) peer_id err:(CFErrorRef *)error;
-(CFSetRef) copyPeerSetMatching:(SOSModifyPeerBlock)block;
-(CFArrayRef) copyPeersToListenTo:(SecKeyRef)userPublic err:(CFErrorRef *)error;
-(bool) peerSignatureUpdate:(SecKeyRef)privKey err:(CFErrorRef *)error;
-(bool) updatePeerInfo:(SOSKVSCircleStorageTransport*)circleTransport description:(CFStringRef)updateDescription err:(CFErrorRef *)error update:(SOSModifyPeerInfoBlock)block;
-(bool) addEscrowToPeerInfo:(SOSFullPeerInfoRef) myPeer err:(CFErrorRef *)error;

//Views
-(SOSViewResultCode) updateView:(SOSAccount*)account name:(CFStringRef) viewname code:(SOSViewActionCode) actionCode err:(CFErrorRef *)error;
-(SOSViewResultCode) viewStatus:(SOSAccount*)account name:(CFStringRef) viewname err:(CFErrorRef *)error;
-(bool) matchOTUserViewSettings: (SOSAccount*)account userViewsEnabled: (bool) otHasUserViewsEnabled err:(CFErrorRef *)error;
-(bool) updateViewSets:(SOSAccount*)account enabled:(CFSetRef) origEnabledViews disabled:(CFSetRef) origDisabledViews;

-(CFSetRef) copyPeerSetForView:(CFStringRef) viewName;

//DER
-(size_t) getDEREncodedSize:(SOSAccount*)account err:(NSError**)error;
-(uint8_t*) encodeToDER:(SOSAccount*)account err:(NSError**) error start:(const uint8_t*) der end:(uint8_t*)der_end;

//Syncing
-(CFMutableSetRef) CF_RETURNS_RETAINED syncWithPeers:(SOSAccountTransaction*) txn peerIDs:(CFSetRef) /* CFStringRef */ peerIDs err:(CFErrorRef *)error;
-(bool) requestSyncWithAllPeers:(SOSAccountTransaction*) txn key:(SecKeyRef)userPublic err:(CFErrorRef *)error;
-(SOSEngineRef) getDataSourceEngine:(SOSDataSourceFactoryRef)factory;


-(bool) postDebugScope:(SOSCircleStorageTransport*) circle_transport scope:(CFTypeRef) scope err:(CFErrorRef*)error;
-(SecKeyRef) copyDeviceKey:(CFErrorRef *)error;
-(void) addSyncablePeerBlock:(SOSAccountTransaction*)a dsName:(CFStringRef) ds_name change:(SOSAccountSyncablePeersBlock) changeBlock;
-(bool) clientPing:(SOSAccount*)account;
-(void) removeInvalidApplications:(SOSCircleRef) circle userPublic:(SecKeyRef)userPublic;

-(bool) addiCloudIdentity:(SOSCircleRef) circle key:(SecKeyRef) userKey err:(CFErrorRef*)error;
-(bool) removeIncompleteiCloudIdentities:(SOSCircleRef) circle privKey:(SecKeyRef) privKey err:(CFErrorRef *)error;
-(bool) upgradeiCloudIdentity:(SOSCircleRef) circle privKey:(SecKeyRef) privKey;
-(CFMutableSetRef) copyPreApprovedHSA2Info;

-(void) addRingDictionary;
-(void) resetRingDictionary;


@end
#endif /* SOSAccountTrustClassic_h */
