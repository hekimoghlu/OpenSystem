/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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
//  SOSAccountCircles.c
//  sec
//

#include "keychain/SecureObjectSync/SOSAccountPriv.h"
#import "keychain/SecureObjectSync/SOSTransport.h"
#import "keychain/SecureObjectSync/SOSTransportKeyParameter.h"
#import "keychain/SecureObjectSync/SOSTransportMessageKVS.h"
#import "keychain/SecureObjectSync/SOSTransportCircleKVS.h"
#import "keychain/SecureObjectSync/SOSTransportCircleCK.h"
#import "keychain/SecureObjectSync/SOSAccountTrust.h"
#import "keychain/SecureObjectSync/SOSAccountTrustClassic.h"
#import "keychain/SecureObjectSync/SOSAccountTrustClassic+Expansion.h"
#import "keychain/SecureObjectSync/SOSAccountTrustClassic+Circle.h"

#include "keychain/SecureObjectSync/SOSPeerInfoCollections.h"
#include "SOSCloudKeychainClient.h"

//
// MARK: Circle management
//


SOSCircleRef CF_RETURNS_RETAINED SOSAccountEnsureCircle(SOSAccount* a, CFStringRef name, CFErrorRef *error)
{
    CFErrorRef localError = NULL;
    SOSAccountTrustClassic *trust = a.trust;
    SOSCircleRef circle = trust.trustedCircle;

    if (circle == NULL) {
        circle = SOSCircleCreate(NULL, name, NULL);
        a.key_interests_need_updating = true;
        [trust setTrustedCircle:circle];
        [a sosEvaluateIfNeeded];
    } else {
        CFRetainSafe(circle);
    }

    require_action_quiet(circle || !isSOSErrorCoded(localError, kSOSErrorIncompatibleCircle), fail,
                         if (error) { *error = localError; localError = NULL; });
   
fail:
    CFReleaseNull(localError);
    return circle;
}

bool SOSAccountEvaluateKeysAndCircle(SOSAccountTransaction *txn, CFErrorRef *error) {
    // if the userKey signature on the circle doesn't work with the new userkey
    if([txn.account.trust isInCircleOnly:nil]) {
        return SOSAccountGenerationSignatureUpdate(txn.account, error);
    }
    return true;
}
