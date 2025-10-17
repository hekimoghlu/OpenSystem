/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 1, 2023.
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
//  SOSPeerInfoDER.h
//  sec
//
//  Created by Richard Murphy on 2/9/15.
//
//

#ifndef _sec_SOSPeerInfoDER_
#define _sec_SOSPeerInfoDER_

#include <corecrypto/ccder.h>
#include <utilities/der_date.h>
#include <Security/SecureObjectSync/SOSPeerInfo.h>

#include <stdio.h>
//
// DER Import Export
//
SOSPeerInfoRef SOSPeerInfoCreateFromDER(CFAllocatorRef allocator, CFErrorRef* error,
                                        const uint8_t** der_p, const uint8_t *der_end);

SOSPeerInfoRef SOSPeerInfoCreateFromData(CFAllocatorRef allocator, CFErrorRef* error,
                                         CFDataRef peerinfo_data);

size_t      SOSPeerInfoGetDEREncodedSize(SOSPeerInfoRef peer, CFErrorRef *error);
uint8_t*    SOSPeerInfoEncodeToDER(SOSPeerInfoRef peer, CFErrorRef* error,
                                   const uint8_t* der, uint8_t* der_end);

CFDataRef SOSPeerInfoCopyEncodedData(SOSPeerInfoRef peer, CFAllocatorRef allocator, CFErrorRef *error);

#endif /* defined(_sec_SOSPeerInfoDER_) */
