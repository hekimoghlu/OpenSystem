/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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
#ifndef _DER_PLIST_H_
#define _DER_PLIST_H_

#include <CoreFoundation/CoreFoundation.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// Error Codes for PropertyList <-> DER
//

static const CFIndex kSecDERErrorUnknownEncoding = -1;
static const CFIndex kSecDERErrorUnsupportedDERType = -2;
static const CFIndex kSecDERErrorAllocationFailure = -3;
static const CFIndex kSecDERErrorUnsupportedNumberType = -4;
static const CFIndex kSecDERErrorUnsupportedCFObject = -5;
static const CFIndex kSecDERErrorNullInput = -6;
static const CFIndex kSecDERErrorCCDEREncode = -7;
static const CFIndex kSecDERErrorOverflow = -8;

extern CFStringRef sSecDERErrorDomain;

enum {
    kCFPropertyListDERFormat_v1_0 = 400
};


// PropertyList <-> DER Functions

size_t der_sizeof_plist(CFPropertyListRef pl, CFErrorRef *error);

uint8_t* der_encode_plist(CFPropertyListRef pl, CFErrorRef *error,
                           const uint8_t *der, uint8_t *der_end);

// When allowed to repair, if certain objects (right now only Dates) do not validate, set them to recognizable defaults
uint8_t* der_encode_plist_repair(CFPropertyListRef pl, CFErrorRef *error,
                                 bool repair, const uint8_t *der, uint8_t *der_end);

const uint8_t* der_decode_plist(CFAllocatorRef pl,
                                CFPropertyListRef* cf, CFErrorRef *error,
                                const uint8_t* der, const uint8_t *der_end);

CFDataRef CFPropertyListCreateDERData(CFAllocatorRef allocator, CFPropertyListRef plist, CFErrorRef *error);

CFPropertyListRef CFPropertyListCreateWithDERData(CFAllocatorRef allocator, CFDataRef data, CFOptionFlags options, CFPropertyListFormat *format, CFErrorRef *error);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
