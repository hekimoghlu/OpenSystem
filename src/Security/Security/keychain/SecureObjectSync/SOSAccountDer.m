/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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
#include "SOSAccountPriv.h"
#include "keychain/SecureObjectSync/SOSCircleDer.h"

//
// DER Encoding utilities
//

static const uint8_t* ccder_decode_null(const uint8_t* der, const uint8_t *der_end)
{
    if (NULL == der)
        return NULL;

    size_t payload_size = 0;
    const uint8_t *payload = ccder_decode_tl(CCDER_NULL, &payload_size, der, der_end);

    if (NULL == payload || payload_size != 0) {
        return NULL;
    }

    return payload + payload_size;
}


static size_t ccder_sizeof_null(void)
{
    return ccder_sizeof(CCDER_NULL, 0);
}


static uint8_t* ccder_encode_null(const uint8_t *der, uint8_t *der_end)
{
    return ccder_encode_tl(CCDER_NULL, 0, der, der_end);
}


//
// Encodes data or a zero length data
//
size_t der_sizeof_fullpeer_or_null(SOSFullPeerInfoRef full_peer, CFErrorRef* error)
{
    if (full_peer) {
        return SOSFullPeerInfoGetDEREncodedSize(full_peer, error);
    } else {
        return ccder_sizeof_null();
    }
}

uint8_t* der_encode_fullpeer_or_null(SOSFullPeerInfoRef full_peer, CFErrorRef* error, const uint8_t* der, uint8_t* der_end)
{
    if (full_peer) {
        return SOSFullPeerInfoEncodeToDER(full_peer, error, der, der_end);
    } else {
        return ccder_encode_null(der, der_end);
    }
}


const uint8_t* der_decode_fullpeer_or_null(CFAllocatorRef allocator, SOSFullPeerInfoRef* full_peer,
                                           CFErrorRef* error,
                                           const uint8_t* der, const uint8_t* der_end)
{
    ccder_tag tag;

    require_action_quiet(ccder_decode_tag(&tag, der, der_end), fail, der = NULL);

    require_action_quiet(full_peer, fail, der = NULL);

    if (tag == CCDER_NULL) {
        der = ccder_decode_null(der, der_end);
    } else  {
        *full_peer = SOSFullPeerInfoCreateFromDER(kCFAllocatorDefault, error, &der, der_end);
    }

fail:
    return der;
}


//
// Mark: public_bytes encode/decode
//

size_t der_sizeof_public_bytes(SecKeyRef publicKey, CFErrorRef* error)
{
    CFDataRef publicData = NULL;
    
    if (publicKey)
        SecKeyCopyPublicBytes(publicKey, &publicData);
    
    size_t size = der_sizeof_data_or_null(publicData, error);
    
    CFReleaseNull(publicData);
    
    return size;
}

uint8_t* der_encode_public_bytes(SecKeyRef publicKey, CFErrorRef* error, const uint8_t* der, uint8_t* der_end)
{
    CFDataRef publicData = NULL;
    
    if (publicKey)
        SecKeyCopyPublicBytes(publicKey, &publicData);
    
    uint8_t *result = der_encode_data_or_null(publicData, error, der, der_end);
    
    CFReleaseNull(publicData);
    
    return result;
}

const uint8_t* der_decode_public_bytes(CFAllocatorRef allocator, CFIndex algorithmID, SecKeyRef* publicKey, CFErrorRef* error, const uint8_t* der, const uint8_t* der_end)
{
    CFDataRef dataFound = NULL;
    der = der_decode_data_or_null(allocator, &dataFound, error, der, der_end);
    
    if (der && dataFound && publicKey) {
        *publicKey = SecKeyCreateFromPublicData(allocator, algorithmID, dataFound);
    }
    CFReleaseNull(dataFound);
    
    return der;
}

