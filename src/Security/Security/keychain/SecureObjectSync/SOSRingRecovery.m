/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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
//  SOSRingRecovery.c
//  sec
//

#include "SOSRingRecovery.h"
#include "SOSRingBackup.h"

#include <AssertMacros.h>

#include "keychain/SecureObjectSync/SOSInternal.h"
#include "keychain/SecureObjectSync/SOSPeerInfoInternal.h"
#include "keychain/SecureObjectSync/SOSPeerInfoCollections.h"
#include "keychain/SecureObjectSync/SOSCircle.h"
#include <Security/SecureObjectSync/SOSViews.h>
#include "keychain/SecureObjectSync/SOSRecoveryKeyBag.h"

#include <Security/SecFramework.h>

#include <Security/SecKey.h>
#include <Security/SecKeyPriv.h>
#include <CoreFoundation/CoreFoundation.h>

#include <utilities/SecCFWrappers.h>

#include <stdlib.h>

#include "SOSRingUtils.h"
#include "SOSRingTypes.h"
#include "SOSRingBasic.h"

// MARK: Recovery Ring Ops

static SOSRingRef SOSRingCreate_Recovery(CFStringRef name, CFStringRef myPeerID, CFErrorRef *error) {
    return SOSRingCreate_ForType(name, kSOSRingRecovery, myPeerID, error);
}



ringFuncStruct recovery = {
    "Recovery",
    1,
    SOSRingCreate_Recovery,
    SOSRingResetToEmpty_Basic,
    SOSRingResetToOffering_Basic,
    SOSRingDeviceIsInRing_Basic,
    SOSRingApply_Basic,
    SOSRingWithdraw_Basic,
    SOSRingGenerationSign_Basic,
    SOSRingConcordanceSign_Basic,
    SOSRingPeerKeyConcordanceTrust,
    NULL,
    NULL,
    SOSRingSetPayload_Basic,
    SOSRingGetPayload_Basic,
};


static bool isRecoveryRing(SOSRingRef ring, CFErrorRef *error) {
    SOSRingType type = SOSRingGetType(ring);
    require_quiet(kSOSRingRecovery == type, errOut);
    return true;
errOut:
    SOSCreateError(kSOSErrorUnexpectedType, CFSTR("Not recovery ring type"), NULL, error);
    return false;
}

bool SOSRingSetRecoveryKeyBag(SOSRingRef ring, SOSFullPeerInfoRef fpi, SOSRecoveryKeyBagRef rkbg, CFErrorRef *error) {
    SOSRingAssertStable(ring);
    CFDataRef rkbg_as_data = NULL;
    bool result = false;
    require_quiet(isRecoveryRing(ring, error), errOut);
    
    rkbg_as_data = SOSRecoveryKeyBagCopyEncoded(rkbg, error);
    result = rkbg_as_data &&
    SOSRingSetPayload(ring, NULL, rkbg_as_data, fpi, error);
errOut:
    CFReleaseNull(rkbg_as_data);
    return result;
}

SOSRecoveryKeyBagRef SOSRingCopyRecoveryKeyBag(SOSRingRef ring, CFErrorRef *error) {
    SOSRingAssertStable(ring);
    
    CFDataRef rkbg_as_data = NULL;
    SOSRecoveryKeyBagRef result = NULL;
    require_quiet(isRecoveryRing(ring, error), errOut);
    
    rkbg_as_data = SOSRingGetPayload(ring, error);
    require_quiet(rkbg_as_data, errOut);
    
    result = SOSRecoveryKeyBagCreateFromData(kCFAllocatorDefault, rkbg_as_data, error);
    
errOut:
    return result;
}
