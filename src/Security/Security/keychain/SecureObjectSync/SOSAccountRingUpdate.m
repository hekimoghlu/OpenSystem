/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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
//  SOSAccountRingUpdate.c
//  sec
//
//

#include <stdio.h>

#include "SOSAccountPriv.h"
#include "keychain/SecureObjectSync/SOSTransportCircleKVS.h"
#include "keychain/SecureObjectSync/SOSTransport.h"
#include <Security/SecureObjectSync/SOSViews.h>
#include "keychain/SecureObjectSync/SOSRing.h"
#include "keychain/SecureObjectSync/SOSRingUtils.h"
#include "keychain/SecureObjectSync/SOSPeerInfoCollections.h"
#import "keychain/SecureObjectSync/SOSAccountTrust.h"

bool SOSAccountIsPeerRetired(SOSAccount* account, CFSetRef peers){
    CFMutableArrayRef peerInfos = CFArrayCreateMutableForCFTypes(kCFAllocatorDefault);
    bool result = false;
    
    CFSetForEach(peers, ^(const void *value) {
        SOSPeerInfoRef peer = (SOSPeerInfoRef)value;
        if(SOSPeerInfoIsRetirementTicket(peer))
            CFArrayAppendValue(peerInfos, peer);
    });
    if(CFArrayGetCount(peerInfos) > 0){
        if(!SOSAccountRemoveBackupPeers(account, peerInfos, NULL))
            secerror("Could not remove peers: %@, from the backup", peerInfos);
        else
            return true;
    }
    else
        result = true;
    
    CFReleaseNull(peerInfos);
    
    return result;
}
