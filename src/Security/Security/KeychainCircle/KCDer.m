/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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
//  KCDer.m
//  Security
//
//

#import <Foundation/Foundation.h>

#include <KeychainCircle/KCDer.h>
#import <KeychainCircle/KCError.h>
#import <os/overflow.h>

// These should probably be shared with security, but we don't export our der'izing functions yet.


static const uint8_t* kcder_decode_data_internal(NSData** data, bool copy,
                                          NSError**error,
                                          const uint8_t* der, const uint8_t *der_end)
{
    if (NULL == der)
        return NULL;

    size_t payload_size = 0;
    const uint8_t *payload = ccder_decode_tl(CCDER_OCTET_STRING, &payload_size, der, der_end);

    uintptr_t payload_end_computed = 0;
    if(os_add_overflow((uintptr_t)payload, payload_size, &payload_end_computed)) {
        KCJoiningErrorCreate(kDERUnknownEncoding, error, @"Bad payload size");
        return NULL;
    }
    if (NULL == payload || payload_end_computed > (uintptr_t) der_end) {
        KCJoiningErrorCreate(kDERUnknownEncoding, error, @"Unknown data encoding");
        return NULL;
    }

    *data = copy ? [NSData dataWithBytes: (void*)payload length: payload_size] :
                   [NSData dataWithBytesNoCopy: (void*)payload length:payload_size freeWhenDone:NO];

    if (NULL == *data) {
        KCJoiningErrorCreate(kAllocationFailure, error, @"Allocation failure!");
        return NULL;
    }

    return payload + payload_size;
}


const uint8_t* kcder_decode_data_nocopy(NSData** data,
                                        NSError**error,
                                        const uint8_t* der, const uint8_t *der_end)
{
    return kcder_decode_data_internal(data, NO, error, der, der_end);
}

const uint8_t* kcder_decode_data(NSData** data,
                                 NSError**error,
                                 const uint8_t* der, const uint8_t *der_end) {
    return kcder_decode_data_internal(data, YES, error, der, der_end);
}


size_t kcder_sizeof_data(NSData* data, NSError** error) {
    return ccder_sizeof_raw_octet_string(data.length);
}

uint8_t* kcder_encode_data_optional(NSData* _Nullable data, NSError**error,
                           const uint8_t *der, uint8_t *der_end)
{
    if (data == nil) return der_end;

    return kcder_encode_data(data, error, der, der_end);

}


uint8_t* kcder_encode_data(NSData* data, NSError**error,
                                  const uint8_t *der, uint8_t *der_end)
{
    return ccder_encode_tl(CCDER_OCTET_STRING, data.length, der,
                           ccder_encode_body(data.length, data.bytes, der, der_end));

}


const uint8_t* kcder_decode_string(NSString** string, NSError**error,
                                          const uint8_t* der, const uint8_t *der_end)
{
    if (NULL == der)
        return NULL;

    size_t payload_size = 0;
    const uint8_t *payload = ccder_decode_tl(CCDER_UTF8_STRING, &payload_size, der, der_end);

    uintptr_t payload_end_computed = 0;
    if(os_add_overflow((uintptr_t)payload, payload_size, &payload_end_computed)) {
        KCJoiningErrorCreate(kDERUnknownEncoding, error, @"Bad payload size");
        return NULL;
    }
    if (NULL == payload || payload_end_computed > (uintptr_t) der_end) {
        KCJoiningErrorCreate(kDERUnknownEncoding, error, @"Unknown string encoding");
        return NULL;
    }

    *string = [[NSString alloc] initWithBytes:payload length:payload_size encoding:NSUTF8StringEncoding];

    if (nil == *string) {
        KCJoiningErrorCreate(kAllocationFailure, error, @"Allocation failure!");
        return NULL;
    }

    return payload + payload_size;
}


size_t kcder_sizeof_string(NSString* string, NSError** error)
{
    return ccder_sizeof(CCDER_UTF8_STRING, [string lengthOfBytesUsingEncoding:NSUTF8StringEncoding]);
}


uint8_t* kcder_encode_string(NSString* string, NSError** error,
                                    const uint8_t *der, uint8_t *der_end)
{
    // Obey the NULL allowed rules.
    if (!der_end)
        return NULL;

    NSUInteger max = (der_end - der);
    void *buffer = der_end - max;
    NSUInteger used = 0;
    if (![string getBytes:buffer
                maxLength:max
               usedLength:&used
                 encoding:NSUTF8StringEncoding
                  options:0
                    range:NSMakeRange(0, string.length)
           remainingRange:nil]) {
        KCJoiningErrorCreate(kDERStringEncodingFailed, error, @"String encoding failed");
        return NULL;
    }

    return ccder_encode_tl(CCDER_UTF8_STRING, used, der,
                           ccder_encode_body(used, buffer, der, der_end));
    
}

uint8_t *kcder_encode_raw_octet_space(size_t s_size, uint8_t **location,
                                      const uint8_t *der, uint8_t *der_end) {
    der_end = ccder_encode_body_nocopy(s_size, der, der_end);
    if (der_end && location)
        *location = der_end;

    return ccder_encode_tl(CCDER_OCTET_STRING, s_size, der, der_end);
}

