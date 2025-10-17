/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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

#include <CoreFoundation/CoreFoundation.h>
#include "keychain/SecureObjectSync/SOSTransportBackupPeer.h"
#include <utilities/SecCFWrappers.h>
#include <AssertMacros.h>


CFGiblisWithHashFor(SOSTransportBackupPeer);

SOSTransportBackupPeerRef SOSTransportBackupPeerCreate(CFStringRef fileLocation, CFErrorRef *error)
{
    SOSTransportBackupPeerRef tpt = (SOSTransportBackupPeerRef)CFTypeAllocateWithSpace(SOSTransportBackupPeer, sizeof(struct __OpaqueSOSTransportBackupPeer) - sizeof(CFRuntimeBase), kCFAllocatorDefault);
    tpt->fileLocation = CFRetainSafe(fileLocation);
    return tpt;
}

static CFStringRef SOSTransportBackupPeerCopyFormatDescription(CFTypeRef aObj, CFDictionaryRef formatOptions){
    SOSTransportBackupPeerRef t = (SOSTransportBackupPeerRef) aObj;
    
    return CFStringCreateWithFormat(NULL, NULL, CFSTR("<SOSTransportBackupPeer@%p\n>"), t);
}

static void SOSTransportBackupPeerDestroy(CFTypeRef aObj){
    SOSTransportBackupPeerRef transport = (SOSTransportBackupPeerRef) aObj;
    CFReleaseNull(transport);
   }

CFIndex SOSTransportBackupPeerGetTransportType(SOSTransportBackupPeerRef transport, CFErrorRef *error){
    return 3;
}

static CFHashCode SOSTransportBackupPeerHash(CFTypeRef obj){
    return (intptr_t) obj;
}

static Boolean SOSTransportBackupPeerCompare(CFTypeRef lhs, CFTypeRef rhs){
    return SOSTransportBackupPeerHash(lhs) == SOSTransportBackupPeerHash(rhs);
}
