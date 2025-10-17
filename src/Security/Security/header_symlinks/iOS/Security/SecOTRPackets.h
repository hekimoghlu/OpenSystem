/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
#include <Security/SecOTRSession.h>

#include <CoreFoundation/CFData.h>

#ifndef _SECOTRPACKETS_H_
#define _SECOTRPACKETS_H_

void SecOTRAppendDHMessage(SecOTRSessionRef session, CFMutableDataRef appendTo);
void SecOTRAppendDHKeyMessage(SecOTRSessionRef session, CFMutableDataRef appendTo);
void SecOTRAppendRevealSignatureMessage(SecOTRSessionRef session, CFMutableDataRef appendTo);
void SecOTRAppendSignatureMessage(SecOTRSessionRef session, CFMutableDataRef appendTo);

typedef enum {
    kDHMessage = 0x02,
    kDataMessage = 0x03,
    kDHKeyMessage = 0x0A,
    kRevealSignatureMessage = 0x11,
    kSignatureMessage = 0x12,

    kEvenCompactDataMessage = 0x20,
    kOddCompactDataMessage = 0x21,
    
    kEvenCompactDataMessageWithHashes = 0x30,
    kOddCompactDataMessageWithHashes = 0x31,

    kInvalidMessage = 0xFF
} OTRMessageType;

#endif
