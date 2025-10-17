/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 19, 2024.
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
//  SOSConcordanceTrust.h
//  sec
//
//  Created by Richard Murphy on 3/15/15.
//
//

#ifndef _sec_SOSConcordanceTrust_
#define _sec_SOSConcordanceTrust_

#include <CoreFoundation/CoreFoundation.h>

typedef CF_ENUM(uint32_t, SOSConcordanceStatus) {
    kSOSConcordanceTrusted = 0,
    kSOSConcordanceGenOld = 1,     // kSOSErrorReplay
    kSOSConcordanceNoUserSig = 2,  // kSOSErrorBadSignature
    kSOSConcordanceNoUserKey = 3,  // kSOSErrorNoKey
    kSOSConcordanceNoPeer = 4,     // kSOSErrorPeerNotFound
    kSOSConcordanceBadUserSig = 5, // kSOSErrorBadSignature
    kSOSConcordanceBadPeerSig = 6, // kSOSErrorBadSignature
    kSOSConcordanceNoPeerSig = 7,
    kSOSConcordanceWeSigned = 8,
    kSOSConcordanceInvalidMembership = 9, // Only used for BackupRings so far
    kSOSConcordanceMissingMe = 10, // Only used for BackupRings so far
    kSOSConcordanceImNotWorthy = 11, // Only used for BackupRings so far
    kSOSConcordanceError = 99,
};

#endif /* defined(_sec_SOSConcordanceTrust_) */
