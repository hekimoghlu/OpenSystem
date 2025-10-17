/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
#import <Foundation/NSXPCConnection.h>
#import <objc/runtime.h>
#import <Security/SecXPCHelper.h>
#include <utilities/debugging.h>

#import "keychain/SecureObjectSync/SOSTypes.h"
#import "keychain/SecureObjectSync/SOSControlHelper.h"

void
_SOSControlSetupInterface(NSXPCInterface *interface)
{
    NSSet<Class> *errClasses = [SecXPCHelper safeErrorClasses];

    @try {
        [interface setClasses:errClasses forSelector:@selector(userPublicKey:) argumentIndex:2 ofReply:YES];

        [interface setClasses:errClasses forSelector:@selector(stashedCredentialPublicKey:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(assertStashedAccountCredential:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(validatedStashedAccountCredential:flowID:deviceSessionID:canSendMetrics:complete:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(stashAccountCredential:altDSID:flowID:deviceSessionID:canSendMetrics:complete:) argumentIndex:1 ofReply:YES];

        [interface setClasses:errClasses forSelector:@selector(ghostBust:complete:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(ghostBustPeriodic:complete:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(ghostBustTriggerTimed:complete:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(ghostBustInfo:) argumentIndex:0 ofReply:YES];

        [interface setClasses:errClasses forSelector:@selector(iCloudIdentityStatus:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(accountStatus:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(keyStatusFor:complete:) argumentIndex:1 ofReply:YES];

        [interface setClasses:errClasses forSelector:@selector(myPeerInfo:flowID:deviceSessionID:canSendMetrics:complete:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(circleHash:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(circleJoiningBlob:flowID:deviceSessionID:canSendMetrics:applicant:complete:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(joinCircleWithBlob:altDSID:flowID:deviceSessionID:canSendMetrics:version:complete:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(initialSyncCredentials:altDSID:flowID:deviceSessionID:canSendMetrics:complete:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(importInitialSyncCredentials:complete:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(rpcTriggerSync:complete:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(getWatchdogParameters:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(setWatchdogParmeters:complete:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(rpcTriggerBackup:complete:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(rpcTriggerRingUpdate:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(removeV0Peers:) argumentIndex:1 ofReply:YES];
    }
    @catch(NSException* e) {
        secerror("Could not configure SOSControlHelper: %@", e);
        @throw e;
    }
}
