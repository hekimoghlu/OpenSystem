/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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
//  SOSPeerInfoPriv.h
//  sec
//
//  Created by Richard Murphy on 12/4/14.
//
//

#ifndef sec_SOSPeerInfoPriv_h
#define sec_SOSPeerInfoPriv_h

#include <CoreFoundation/CFRuntime.h>
#include <CoreFoundation/CoreFoundation.h>
#include <utilities/SecCFWrappers.h>

struct __OpaqueSOSPeerInfo {
    CFRuntimeBase           _base;
    //
    CFMutableDictionaryRef  description;
    CFDataRef               signature;
    
    // Cached data
    CFDictionaryRef         gestalt;
    CFStringRef             peerID;
    CFStringRef             spid;
    CFIndex                 version;
    CFStringRef             verifiedAppKeyID;
    bool                    verifiedResult;

    /* V2 and beyond are listed below */
    bool                    v2DictionaryIsExpanded;
    CFMutableDictionaryRef  v2Dictionary;
};

CF_RETURNS_RETAINED SOSPeerInfoRef SOSPeerInfoAllocate(CFAllocatorRef allocator);
bool SOSPeerInfoSign(SecKeyRef privKey, SOSPeerInfoRef peer, CFErrorRef *error);
bool SOSPeerInfoVerify(SOSPeerInfoRef peer, CFErrorRef *error);
void SOSPeerInfoSetVersionNumber(SOSPeerInfoRef pi, int version);

SOSPeerInfoRef SOSPeerInfoCopyWithModification(CFAllocatorRef allocator, SOSPeerInfoRef original,
                                               SecKeyRef signingKey, CFErrorRef *error,
                                               bool (^modification)(SOSPeerInfoRef peerToModify, CFErrorRef *error));

extern const CFStringRef peerIDLengthKey;

#endif
