/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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


const uint8_t* der_decode_boolean(CFAllocatorRef allocator,
                                  CFBooleanRef* boolean, CFErrorRef *error,
                                  const uint8_t* der, const uint8_t *der_end)
{
    if (NULL == der) {
        SecCFDERCreateError(kSecDERErrorNullInput, CFSTR("null input"), NULL, error);
        return NULL;
    }

    size_t payload_size = 0;
    const uint8_t *payload = ccder_decode_tl(CCDER_BOOLEAN, &payload_size, der, der_end);

    if (NULL == payload || (ssize_t) (der_end - payload) < (ssize_t) payload_size || payload_size != 1) {
        SecCFDERCreateError(kSecDERErrorUnknownEncoding, CFSTR("Unknown boolean encoding"), NULL, error);
        return NULL;
    }

    *boolean = *payload ? kCFBooleanTrue : kCFBooleanFalse;

    return payload + payload_size;
}


size_t der_sizeof_boolean(CFBooleanRef data __unused, CFErrorRef *error)
{
    return ccder_sizeof(CCDER_BOOLEAN, 1);
}


uint8_t* der_encode_boolean(CFBooleanRef boolean, CFErrorRef *error,
                            const uint8_t *der, uint8_t *der_end)
{
    // this mis-encoding is load-bearing for SOS, see rdar://132436820
    uint8_t value = CFBooleanGetValue(boolean);

    return SecCCDEREncodeHandleResult(ccder_encode_tl(CCDER_BOOLEAN, 1, der,
                                                      ccder_encode_body(1, &value, der, der_end)),
                                      error);

}
