/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
#include "Security_regressions.h"

#include <Security/SecOTRMath.h>
#include <Security/SecOTRDHKey.h>
#include <utilities/SecCFWrappers.h>

int otr_otrdh(int argc, char *const * argv)
{
    plan_tests(7);
    
    SecOTRFullDHKeyRef aliceFull = SecOTRFullDHKCreate(kCFAllocatorDefault);
    SecOTRPublicDHKeyRef alicePublic = SecOTRPublicDHKCreateFromFullKey(kCFAllocatorDefault, aliceFull);
    
    SecOTRFullDHKeyRef bobFull = SecOTRFullDHKCreate(kCFAllocatorDefault);
    SecOTRPublicDHKeyRef bobPublic = SecOTRPublicDHKCreateFromFullKey(kCFAllocatorDefault, bobFull);

    SecOTRPublicDHKeyRef aliceCompactDeserialized = NULL;
    SecOTRPublicDHKeyRef aliceDeserialized = NULL;

    CFMutableDataRef aliceCompactSerialized = CFDataCreateMutable(kCFAllocatorDefault, 0);
    SecFDHKAppendCompactPublicSerialization(aliceFull, aliceCompactSerialized);
    size_t aliceCompactLength = CFDataGetLength(aliceCompactSerialized);
    const uint8_t *aliceCompactSerializationStart = CFDataGetMutableBytePtr(aliceCompactSerialized);

    aliceCompactDeserialized = SecOTRPublicDHKCreateFromCompactSerialization(kCFAllocatorDefault, &aliceCompactSerializationStart, &aliceCompactLength);
    ok(CFEqualSafe(aliceCompactDeserialized, alicePublic), "Compact serialized compare to created");


    CFMutableDataRef aliceSerialized = CFDataCreateMutable(kCFAllocatorDefault, 0);
    SecFDHKAppendPublicSerialization(aliceFull, aliceSerialized);
    size_t aliceLength = CFDataGetLength(aliceSerialized);
    const uint8_t *aliceSerializationStart = CFDataGetMutableBytePtr(aliceSerialized);

    aliceDeserialized = SecOTRPublicDHKCreateFromSerialization(kCFAllocatorDefault, &aliceSerializationStart, &aliceLength);
    ok(CFEqualSafe(aliceDeserialized, alicePublic), "Serialized compare to created");

    ok(CFEqualSafe(aliceCompactDeserialized, aliceDeserialized), "Serialized compared to compact serailized");

    uint8_t aliceMessageKeys[2][kOTRMessageKeyBytes];
    uint8_t aliceMacKeys[2][kOTRMessageMacKeyBytes];
    
    SecOTRDHKGenerateOTRKeys(aliceFull, bobPublic,
                          aliceMessageKeys[0], aliceMacKeys[0],
                          aliceMessageKeys[1], aliceMacKeys[1]);
    
    uint8_t bobMessageKeys[2][kOTRMessageKeyBytes];
    uint8_t bobMacKeys[2][kOTRMessageMacKeyBytes];
    
    SecOTRDHKGenerateOTRKeys(bobFull, alicePublic,
                          bobMessageKeys[0], bobMacKeys[0],
                          bobMessageKeys[1], bobMacKeys[1]);
    
    
    ok(0 == memcmp(aliceMessageKeys[0], bobMessageKeys[1], sizeof(aliceMessageKeys[0])), "Mac Keys don't match!!");
    ok(0 == memcmp(aliceMessageKeys[1], bobMessageKeys[0], sizeof(aliceMessageKeys[1])), "Mac Keys don't match!!");
    ok(0 == memcmp(aliceMacKeys[0], bobMacKeys[1], sizeof(aliceMacKeys[0])), "Mac Keys don't match!!");
    ok(0 == memcmp(aliceMacKeys[1], bobMacKeys[0], sizeof(aliceMacKeys[1])), "Mac Keys don't match!!");
    CFReleaseNull(aliceCompactSerialized);
    CFReleaseNull(aliceCompactDeserialized);
    CFReleaseNull(aliceSerialized);
    CFReleaseNull(aliceDeserialized);
    CFReleaseNull(aliceFull);
    CFReleaseNull(alicePublic);
    CFReleaseNull(bobFull);
    CFReleaseNull(bobPublic);
    return 0;
}
