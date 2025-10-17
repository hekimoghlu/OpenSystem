/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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
#ifndef _SECOTR_H_
#define _SECOTR_H_

/*
 * Message Protection interfaces
*/

#include <CoreFoundation/CFBase.h>
#include <CoreFoundation/CFData.h>
#include <CoreFoundation/CFError.h>
#include <Security/SecKey.h>

#include <stdint.h>

__BEGIN_DECLS

/*!
    @typedef 
    @abstract   Full identity (public and private) for Message Protection
    @discussion Abstracts what kind of crypto is going on beyond it being public/priate
*/
typedef struct _SecOTRFullIdentity* SecOTRFullIdentityRef;

/*!
     @typedef 
     @abstract   Public identity for Message Protection message validation and encryption to send
     @discussion Abstracts what kind of crypto is going on beyond it being public/priate
 */
typedef struct _SecOTRPublicIdentity* SecOTRPublicIdentityRef;

/*
 * Full identity functions
 */
SecOTRFullIdentityRef SecOTRFullIdentityCreate(CFAllocatorRef allocator, CFErrorRef *error);

// This variant is used by MessageProtection that doesn't use persistent references anymore.
SecOTRFullIdentityRef SecOTRFullIdentityCreateFromSecKeyRef(CFAllocatorRef allocator, SecKeyRef privateKey,
                                                            CFErrorRef *error);

// This variant is used by SOS, and still relies on privateKey having a persistent reference.
SecOTRFullIdentityRef SecOTRFullIdentityCreateFromSecKeyRefSOS(CFAllocatorRef allocator, SecKeyRef privateKey,
                                                                CFErrorRef *error);
SecOTRFullIdentityRef SecOTRFullIdentityCreateFromData(CFAllocatorRef allocator, CFDataRef serializedData, CFErrorRef *error);
    
SecOTRFullIdentityRef SecOTRFullIdentityCreateFromBytes(CFAllocatorRef allocator, const uint8_t**bytes, size_t *size, CFErrorRef *error);

bool SecOTRFIPurgeFromKeychain(SecOTRFullIdentityRef thisID, CFErrorRef *error);

bool SecOTRFIAppendSerialization(SecOTRFullIdentityRef fullID, CFMutableDataRef serializeInto, CFErrorRef *error);


bool SecOTRFIPurgeAllFromKeychain(CFErrorRef *error);


/*
 * Public identity functions
 */
SecOTRPublicIdentityRef SecOTRPublicIdentityCopyFromPrivate(CFAllocatorRef allocator, SecOTRFullIdentityRef fullID, CFErrorRef *error);
    
SecOTRPublicIdentityRef SecOTRPublicIdentityCreateFromSecKeyRef(CFAllocatorRef allocator, SecKeyRef publicKey,
                                                                    CFErrorRef *error);
    
SecOTRPublicIdentityRef SecOTRPublicIdentityCreateFromData(CFAllocatorRef allocator, CFDataRef serializedData, CFErrorRef *error);
SecOTRPublicIdentityRef SecOTRPublicIdentityCreateFromBytes(CFAllocatorRef allocator, const uint8_t**bytes, size_t * size, CFErrorRef *error);

bool SecOTRPIAppendSerialization(SecOTRPublicIdentityRef publicID, CFMutableDataRef serializeInto, CFErrorRef *error);

void SecOTRAdvertiseHashes(bool advertise);
    
__END_DECLS
        
#endif
