/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 31, 2024.
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
#import "keychain/ot/OTClique.h"
#import "keychain/ot/OTControlProtocol.h"
#import "keychain/ot/OTDefines.h"
#import "keychain/ot/OTJoiningConfiguration.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperSpecificUser.h"
#import <Security/SecXPCHelper.h>
#include <utilities/debugging.h>

NSXPCInterface* OTSetupControlProtocol(NSXPCInterface* interface) {
#if OCTAGON
    NSSet<Class> *errorClasses = [SecXPCHelper safeErrorClasses];

    @try {
        [interface setClasses:errorClasses forSelector:@selector(appleAccountSignedIn:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(appleAccountSignedOut:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(notifyIDMSTrustLevelChangeForAltDSID:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(rpcEpochWithArguments:configuration:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(rpcPrepareIdentityAsApplicantWithArguments:configuration:reply:) argumentIndex:5 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(rpcVoucherWithArguments:configuration:peerID:permanentInfo:permanentInfoSig:stableInfo:stableInfoSig:maxCapability:reply:) argumentIndex:2 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(rpcJoinWithArguments:configuration:vouchData:vouchSig:reply:) argumentIndex:0 ofReply:YES];
        [interface setXPCType:XPC_TYPE_FD forSelector:@selector(status:xpcFd:reply:) argumentIndex:1 ofReply:NO];
        [interface setClasses:errorClasses forSelector:@selector(status:xpcFd:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(status:xpcFd:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(fetchEgoPeerID:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(fetchCliqueStatus:configuration:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(fetchTrustStatus:configuration:reply:) argumentIndex:4 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(startOctagonStateMachine:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(resetAndEstablish:resetReason:idmsTargetContext:idmsCuttlefishPassword:notifyIdMS:accountSettings:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(establish:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(leaveClique:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(removeFriendsInClique:peerIDs:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(peerDeviceNamesByPeerID:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(fetchAllViableBottles:source:reply:) argumentIndex:2 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(restoreFromBottle:entropy:bottleID:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(fetchEscrowContents:reply:) argumentIndex:3 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(createRecoveryKey:recoveryKey:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(joinWithRecoveryKey:recoveryKey:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(healthCheck:skipRateLimitingCheck:repair:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(simulateReceivePush:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(waitForOctagonUpgrade:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(postCDPFollowupResult:success:type:error:reply:) argumentIndex:3 ofReply:NO];
        [interface setClasses:errorClasses forSelector:@selector(postCDPFollowupResult:success:type:error:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(tapToRadar:description:radar:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(refetchCKKSPolicy:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(setCDPEnabled:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(getCDPStatus:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(fetchEscrowRecords:source:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(setUserControllableViewsSyncStatus:enabled:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(fetchUserControllableViewsSyncStatus:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(resetAccountCDPContents:idmsTargetContext:idmsCuttlefishPassword:notifyIdMS:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(setLocalSecureElementIdentity:secureElementIdentity:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(removeLocalSecureElementIdentityPeerID:secureElementIdentityPeerID:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(fetchTrustedSecureElementIdentities:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(setAccountSetting:setting:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(fetchAccountSettings:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(fetchAccountWideSettingsWithForceFetch:arguments:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(waitForPriorityViewKeychainDataRecovery:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(createCustodianRecoveryKey:uuid:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(joinWithCustodianRecoveryKey:custodianRecoveryKey:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(preflightJoinWithCustodianRecoveryKey:custodianRecoveryKey:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(removeCustodianRecoveryKey:uuid:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(createInheritanceKey:uuid:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(generateInheritanceKey:uuid:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(storeInheritanceKey:ik:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(joinWithInheritanceKey:inheritanceKey:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(preflightJoinWithInheritanceKey:inheritanceKey:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(removeInheritanceKey:uuid:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(tlkRecoverabilityForEscrowRecordData:recordData:source:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(setMachineIDOverride:machineID:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(isRecoveryKeySet:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(recoverWithRecoveryKey:recoveryKey:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(removeRecoveryKey:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(preflightRecoverOctagonUsingRecoveryKey:recoveryKey:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(clearCliqueFromAccount:resetReason:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(performCKServerUnreadableDataRemoval:altDSID:reply:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(totalTrustedPeers:reply:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errorClasses forSelector:@selector(areRecoveryKeysDistrusted:reply:) argumentIndex:1 ofReply:YES];
    }
    @catch(NSException* e) {
        secerror("OTSetupControlProtocol failed, continuing, but you might crash later: %@", e);
        @throw e;
    }
#endif
    
    return interface;
}
