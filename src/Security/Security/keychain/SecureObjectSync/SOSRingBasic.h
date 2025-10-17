/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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
//  SOSRingBasic.h
//  sec
//
//  Created by Richard Murphy on 3/3/15.
//
//

#ifndef _sec_SOSRingBasic_
#define _sec_SOSRingBasic_

#include "SOSRingTypes.h"

SOSRingRef SOSRingCreate_Basic(CFStringRef name, CFStringRef myPeerID, CFErrorRef *error);
bool SOSRingResetToEmpty_Basic(SOSRingRef ring, CFStringRef myPeerID, CFErrorRef *error);
SOSRingStatus SOSRingDeviceIsInRing_Basic(SOSRingRef ring, CFStringRef peerID);

bool SOSRingResetToOffering_Basic(SOSRingRef ring, SecKeyRef user_privkey, SOSFullPeerInfoRef requestor, CFErrorRef *error);
bool SOSRingApply_Basic(SOSRingRef ring, SecKeyRef user_pubkey, SOSFullPeerInfoRef requestor, CFErrorRef *error);
bool SOSRingWithdraw_Basic(SOSRingRef ring, SecKeyRef user_privkey, SOSFullPeerInfoRef requestor, CFErrorRef *error);
bool SOSRingGenerationSign_Basic(SOSRingRef ring, SecKeyRef user_privkey, SOSFullPeerInfoRef requestor, CFErrorRef *error);
bool SOSRingConcordanceSign_Basic(SOSRingRef ring, SOSFullPeerInfoRef requestor, CFErrorRef *error);
bool SOSRingSetPayload_Basic(SOSRingRef ring, SecKeyRef user_privkey, CFDataRef payload, SOSFullPeerInfoRef requestor, CFErrorRef *error);
CFDataRef SOSRingGetPayload_Basic(SOSRingRef ring, CFErrorRef *error);

extern ringFuncStruct basic;

#endif /* defined(_sec_SOSRingBasic_) */
