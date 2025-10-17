/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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


const uint8_t* der_decode_string(CFAllocatorRef allocator,
                                 CFStringRef* string, CFErrorRef *error,
                                 const uint8_t* der, const uint8_t *der_end)
{
    if (NULL == der) {
        SecCFDERCreateError(kSecDERErrorNullInput, CFSTR("null input"), NULL, error);
        return NULL;
    }

    size_t payload_size = 0;
    const uint8_t *payload = ccder_decode_tl(CCDER_UTF8_STRING, &payload_size, der, der_end);

    if (NULL == payload || (ssize_t) (der_end - payload) < (ssize_t) payload_size){
        SecCFDERCreateError(kSecDERErrorUnknownEncoding, CFSTR("Unknown string encoding"), NULL, error);
        return NULL;
    }

    *string = CFStringCreateWithBytes(allocator, payload, payload_size, kCFStringEncodingUTF8, false);

    if (NULL == *string) {
        SecCFDERCreateError(kSecDERErrorAllocationFailure, CFSTR("String allocation failed"), NULL, error);
        return NULL;
    }

    return payload + payload_size;
}

const uint8_t* der_decode_numeric_string(CFAllocatorRef allocator,
                                 CFStringRef* string, CFErrorRef *error,
                                 const uint8_t* der, const uint8_t *der_end)
{
    if (NULL == der) {
        SecCFDERCreateError(kSecDERErrorNullInput, CFSTR("null input"), NULL, error);
        return NULL;
    }
    
    size_t payload_size = 0;
    const uint8_t *payload = ccder_decode_tl(CCDER_NUMERIC_STRING, &payload_size, der, der_end);
    
    if (NULL == payload || (ssize_t) (der_end - payload) < (ssize_t) payload_size){
        SecCFDERCreateError(kSecDERErrorUnknownEncoding, CFSTR("Unknown numeric string encoding"), NULL, error);
        return NULL;
    }
    
    *string = CFStringCreateWithBytes(allocator, payload, payload_size, kCFStringEncodingUTF8, false);
    
    if (NULL == *string) {
        SecCFDERCreateError(kSecDERErrorAllocationFailure, CFSTR("Numeric string allocation failed"), NULL, error);
        return NULL;
    }
    
    return payload + payload_size;
}


size_t der_sizeof_string(CFStringRef str, CFErrorRef *error)
{
    const CFIndex str_length    = CFStringGetLength(str);
    const CFIndex maximum       = CFStringGetMaximumSizeForEncoding(str_length, kCFStringEncodingUTF8);

    CFIndex encodedLen = 0;
    CFIndex converted = CFStringGetBytes(str, CFRangeMake(0, str_length), kCFStringEncodingUTF8, 0, false, NULL, maximum, &encodedLen);

    return ccder_sizeof(CCDER_UTF8_STRING, (converted == str_length) ? encodedLen : 0);
}


uint8_t* der_encode_string(CFStringRef string, CFErrorRef *error,
                           const uint8_t *der, uint8_t *der_end)
{
    if (NULL == der_end) {
        SecCFDERCreateError(kSecDERErrorNullInput, CFSTR("null input"), NULL, error);
        return NULL;
    }

    const CFIndex str_length = CFStringGetLength(string);

    ptrdiff_t der_space = der_end - der;
    CFIndex bytes_used = 0;
    uint8_t *buffer = der_end - der_space;
    CFIndex converted = CFStringGetBytes(string, CFRangeMake(0, str_length), kCFStringEncodingUTF8, 0, false, buffer, der_space, &bytes_used);
    if (converted != str_length){
        SecCFDERCreateError(kSecDERErrorUnsupportedCFObject, CFSTR("String extraction failed"), NULL, error);
        return NULL;
    }

    return SecCCDEREncodeHandleResult(ccder_encode_tl(CCDER_UTF8_STRING, bytes_used, der,
                                                      ccder_encode_body(bytes_used, buffer, der, der_end)),
                                      error);

}
