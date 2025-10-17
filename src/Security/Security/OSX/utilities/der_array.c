/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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


const uint8_t* der_decode_array(CFAllocatorRef allocator,
                                CFArrayRef* array, CFErrorRef *error,
                                const uint8_t* der, const uint8_t *der_end)
{
    if (NULL == der) {
        SecCFDERCreateError(kSecDERErrorNullInput, CFSTR("null input"), NULL, error);
        return NULL;
    }

    CFMutableArrayRef result = CFArrayCreateMutable(allocator, 0, &kCFTypeArrayCallBacks);

    const uint8_t *elements_end;
    const uint8_t *current_element = ccder_decode_sequence_tl(&elements_end, der, der_end);
    if (!current_element) {
        SecCFDERCreateError(kSecDERErrorUnknownEncoding, CFSTR("tag/length decode failed"), NULL, error);
    }
    
    while (current_element != NULL && current_element < elements_end) {
        CFPropertyListRef element = NULL;
        current_element = der_decode_plist(allocator, &element, error, current_element, elements_end);
        if (current_element) {
            CFArrayAppendValue(result, element);
            CFReleaseNull(element);
        }
    }

    if (current_element) {
        *array = result;
        result = NULL;
    }

    CFReleaseNull(result);
    return current_element;
}


size_t der_sizeof_array(CFArrayRef data, CFErrorRef *error)
{
    size_t body_size = 0;
    for(CFIndex position = CFArrayGetCount(data) - 1;
        position >= 0;
        --position)
    {
        body_size += der_sizeof_plist(CFArrayGetValueAtIndex(data, position), error);
    }
    return ccder_sizeof(CCDER_CONSTRUCTED_SEQUENCE, body_size);
}


uint8_t* der_encode_array(CFArrayRef array, CFErrorRef *error,
                          const uint8_t *der, uint8_t *der_end)
{
    return der_encode_array_repair(array, error, false, der, der_end);
}

uint8_t* der_encode_array_repair(CFArrayRef array, CFErrorRef *error,
                                 bool repair,
                                 const uint8_t *der, uint8_t *der_end)
{
    uint8_t* original_der_end = der_end;
    for(CFIndex position = CFArrayGetCount(array) - 1;
        position >= 0;
        --position)
    {
        der_end = der_encode_plist_repair(CFArrayGetValueAtIndex(array, position), error, repair, der, der_end);
    }

    return SecCCDEREncodeHandleResult(ccder_encode_constructed_tl(CCDER_CONSTRUCTED_SEQUENCE, original_der_end, der, der_end),
                                      error);
}
