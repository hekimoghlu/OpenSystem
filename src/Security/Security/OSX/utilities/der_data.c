/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
#include <stdio.h>

#include "utilities/SecCFRelease.h"
#include "utilities/der_plist.h"
#include "utilities/der_plist_internal.h"

#include <corecrypto/ccder.h>
#include <CoreFoundation/CoreFoundation.h>
#define CORE_ENTITLEMENTS_I_KNOW_WHAT_IM_DOING
#include <CoreEntitlements/CoreEntitlementsPriv.h>

const uint8_t* der_decode_data(CFAllocatorRef allocator,
                               CFDataRef* data, CFErrorRef *error,
                               const uint8_t* der, const uint8_t *der_end)
{
    if (NULL == der) {
        SecCFDERCreateError(kSecDERErrorNullInput, CFSTR("null input"), NULL, error);
        return NULL;
    }

    size_t payload_size = 0;
    const uint8_t *payload = ccder_decode_tl(CCDER_OCTET_STRING, &payload_size, der, der_end);

    if (NULL == payload || (ssize_t) (der_end - payload) < (ssize_t) payload_size) {
        SecCFDERCreateError(kSecDERErrorUnknownEncoding, CFSTR("Unknown data encoding"), NULL, error);
        return NULL;
    }
    
    *data = CFDataCreate(allocator, payload, payload_size);

    if (NULL == *data) {
        SecCFDERCreateError(kSecDERErrorAllocationFailure, CFSTR("Failed to create data"), NULL, error);
        return NULL;
    }

    return payload + payload_size;
}

const uint8_t* der_decode_core_entitlements_data(CFAllocatorRef allocator,
                                                 CFDataRef* data, CFErrorRef *error,
                                                 const uint8_t* der, const uint8_t *der_end)
{
    if (NULL == der) {
        SecCFDERCreateError(kSecDERErrorNullInput, CFSTR("null input"), NULL, error);
        return NULL;
    }
    
    size_t payload_size = 0;
    const uint8_t *payload = ccder_decode_tl(CCDER_ENTITLEMENTS, &payload_size, der, der_end);
    
    if (NULL == payload || (ssize_t) (der_end - payload) < (ssize_t) payload_size) {
        SecCFDERCreateError(kSecDERErrorUnknownEncoding, CFSTR("Unknown CoreEntitlements encoding"), NULL, error);
        return NULL;
    }
    
    // the entitlements structure is the whole thing
    *data = CFDataCreate(allocator, der, (size_t)((payload + payload_size) - der));
    
    if (NULL == *data) {
        SecCFDERCreateError(kSecDERErrorAllocationFailure, CFSTR("Failed to create CoreEntitlements data"), NULL, error);
        return NULL;
    }
    
    return payload + payload_size;
}


size_t der_sizeof_data(CFDataRef data, CFErrorRef *error)
{
    return ccder_sizeof_raw_octet_string(CFDataGetLength(data));
}


uint8_t* der_encode_data(CFDataRef data, CFErrorRef *error,
                         const uint8_t *der, uint8_t *der_end)
{
    const CFIndex data_length = CFDataGetLength(data);

    return SecCCDEREncodeHandleResult(ccder_encode_tl(CCDER_OCTET_STRING, data_length, der,
                                                      ccder_encode_body(data_length, CFDataGetBytePtr(data), der, der_end)),
                                      error);

}
