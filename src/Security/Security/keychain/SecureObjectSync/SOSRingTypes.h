/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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
//  SOSRingTypes.h
//

#ifndef _sec_SOSRingTypes_
#define _sec_SOSRingTypes_


#include <CoreFoundation/CFRuntime.h>
#include <CoreFoundation/CoreFoundation.h>
#include "keychain/SecureObjectSync/SOSAccount.h"
#include "keychain/SecureObjectSync/SOSRingUtils.h"

typedef struct ringfuncs_t {
    char                    *typeName;
    int                     version;
    SOSRingRef              (*sosRingCreate)(CFStringRef name, CFStringRef myPeerID, CFErrorRef *error);
    bool                    (*sosRingResetToEmpty)(SOSRingRef ring, CFStringRef myPeerID, CFErrorRef *error);
    bool                    (*sosRingResetToOffering)(SOSRingRef ring, SecKeyRef user_privkey, SOSFullPeerInfoRef requestor, CFErrorRef *error);
    SOSRingStatus           (*sosRingDeviceIsInRing)(SOSRingRef ring, CFStringRef peerID);
    bool                    (*sosRingApply)(SOSRingRef ring, SecKeyRef user_pubkey, SOSFullPeerInfoRef requestor, CFErrorRef *error);
    bool                    (*sosRingWithdraw)(SOSRingRef ring, SecKeyRef user_privkey, SOSFullPeerInfoRef requestor, CFErrorRef *error);
    bool                    (*sosRingGenerationSign)(SOSRingRef ring, SecKeyRef user_privkey, SOSFullPeerInfoRef requestor, CFErrorRef *error);
    bool                    (*sosRingConcordanceSign)(SOSRingRef ring, SOSFullPeerInfoRef requestor, CFErrorRef *error);
    SOSConcordanceStatus    (*sosRingConcordanceTrust)(SOSFullPeerInfoRef me, CFSetRef peers,
                                                       SOSRingRef knownRing, SOSRingRef proposedRing,
                                                       SecKeyRef knownPubkey, SecKeyRef userPubkey,
                                                       CFStringRef excludePeerID, CFErrorRef *error);
    bool                    (*sosRingAccept)(SOSRingRef ring, SecKeyRef user_privkey, SOSFullPeerInfoRef requestor, CFErrorRef *error);
    bool                    (*sosRingReject)(SOSRingRef ring, SecKeyRef user_privkey, SOSFullPeerInfoRef requestor, CFErrorRef *error);
    bool                    (*sosRingSetPayload)(SOSRingRef ring, SecKeyRef user_privkey, CFDataRef payload, SOSFullPeerInfoRef requestor, CFErrorRef *error);
    CFDataRef               (*sosRingGetPayload)(SOSRingRef ring, CFErrorRef *error);
} ringFuncStruct, *ringFuncs;

static inline SOSRingRef SOSRingCreate_ForType(CFStringRef name, SOSRingType type, CFStringRef myPeerID, CFErrorRef *error) {
    SOSRingRef retval = NULL;
    retval = SOSRingCreate_Internal(name, type, error);
    if(!retval) return NULL;
    SOSRingSetLastModifier(retval, myPeerID);
    return retval;
}

#endif /* defined(_sec_SOSRingTypes_) */
