/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#include "SecCoreCrypto.h"
#include <corecrypto/ccder.h>


#ifndef CCDER_BOOL_SUPPORT

//
// bool encoding/decoding
//


const uint8_t* ccder_decode_bool(bool* boolean, const uint8_t* der, const uint8_t *der_end)
{
    if (NULL == der)
        return NULL;
    
    size_t payload_size = 0;
    const uint8_t *payload = ccder_decode_tl(CCDER_BOOLEAN, &payload_size, der, der_end);
    
    if (NULL == payload || (der_end - payload) < 1 || payload_size != 1) {
        return NULL;
    }
    
    if (boolean)
        *boolean = (*payload != 0);
    
    return payload + payload_size;
}


size_t ccder_sizeof_bool(bool value __unused, CFErrorRef *error)
{
    return ccder_sizeof(CCDER_BOOLEAN, 1);
}


uint8_t* ccder_encode_bool(bool value, const uint8_t *der, uint8_t *der_end)
{
    uint8_t value_byte = value;
    
    return ccder_encode_tl(CCDER_BOOLEAN, 1, der,
                           ccder_encode_body(1, &value_byte, der, der_end));
}
#endif
