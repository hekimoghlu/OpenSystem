/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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
#ifndef __SOSPEERINFOCOLLECTIONS__
#define __SOSPEERINFOCOLLECTIONS__

#include <Security/SecureObjectSync/SOSPeerInfo.h>
#include <xpc/xpc.h>

//
// CFSet of PeerInfos by ID.
//
extern const CFSetCallBacks kSOSPeerSetCallbacks;
CFMutableSetRef CFSetCreateMutableForSOSPeerInfosByID(CFAllocatorRef allocator);
CFMutableSetRef CFSetCreateMutableForSOSPeerInfosByIDWithArray(CFAllocatorRef allocator, CFArrayRef peerInfos);

bool SOSPeerInfoSetContainsIdenticalPeers(CFSetRef set1, CFSetRef set2);
SOSPeerInfoRef SOSPeerInfoSetFindByID(CFSetRef set, CFStringRef id);

//
// Der encode
//
CFMutableSetRef SOSPeerInfoSetCreateFromArrayDER(CFAllocatorRef allocator, const CFSetCallBacks *callbacks, CFErrorRef* error,
                                                 const uint8_t** der_p, const uint8_t *der_end);
size_t SOSPeerInfoSetGetDEREncodedArraySize(CFSetRef pia, CFErrorRef *error);
uint8_t* SOSPeerInfoSetEncodeToArrayDER(CFSetRef pia, CFErrorRef* error, const uint8_t* der, uint8_t* der_end);

//
// CFArray of Peer Info handling
//

void CFArrayOfSOSPeerInfosSortByID(CFMutableArrayRef peerInfoArray);

//
// Peer Info Array Persistence
//

CFMutableArrayRef SOSPeerInfoArrayCreateFromDER(CFAllocatorRef allocator, CFErrorRef* error,
                                                const uint8_t** der_p, const uint8_t *der_end);
size_t SOSPeerInfoArrayGetDEREncodedSize(CFArrayRef pia, CFErrorRef *error);
uint8_t* SOSPeerInfoArrayEncodeToDER(CFArrayRef pia, CFErrorRef* error, const uint8_t* der, uint8_t* der_end);



CFArrayRef CreateArrayOfPeerInfoWithXPCObject(xpc_object_t peerArray, CFErrorRef* error);
xpc_object_t CreateXPCObjectWithArrayOfPeerInfo(CFArrayRef array, CFErrorRef *error);

#endif
