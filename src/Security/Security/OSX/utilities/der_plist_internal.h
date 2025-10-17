/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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
#ifndef _DER_PLIST_INTERNAL_H_
#define _DER_PLIST_INTERNAL_H_

#include <CoreFoundation/CoreFoundation.h>
#include "utilities/SecCFError.h"

// Always returns false, to satisfy static analysis
#define SecCFDERCreateError(errorCode, descriptionString, previousError, newError) \
    SecCFCreateErrorWithFormat(errorCode, sSecDERErrorDomain, previousError, newError, NULL, descriptionString)

uint8_t * SecCCDEREncodeHandleResult(uint8_t *der, CFErrorRef *newError);


// CFArray <-> DER
size_t der_sizeof_array(CFArrayRef array, CFErrorRef *error);

uint8_t* der_encode_array(CFArrayRef array, CFErrorRef *error,
                          const uint8_t *der, uint8_t *der_end);
uint8_t* der_encode_array_repair(CFArrayRef array, CFErrorRef *error,
                                 bool repair,
                                 const uint8_t *der, uint8_t *der_end);

const uint8_t* der_decode_array(CFAllocatorRef allocator,
                                CFArrayRef* array, CFErrorRef *error,
                                const uint8_t* der, const uint8_t *der_end);

// CFNull <-> DER
size_t der_sizeof_null(CFNullRef	nul, CFErrorRef *error);

uint8_t* der_encode_null(CFNullRef	nul, CFErrorRef *error,
                            const uint8_t *der, uint8_t *der_end);

const uint8_t* der_decode_null(CFAllocatorRef allocator,
                                  CFNullRef	*nul, CFErrorRef *error,
                                  const uint8_t* der, const uint8_t *der_end);


// CFBoolean <-> DER
size_t der_sizeof_boolean(CFBooleanRef boolean, CFErrorRef *error);

uint8_t* der_encode_boolean(CFBooleanRef boolean, CFErrorRef *error,
                            const uint8_t *der, uint8_t *der_end);

const uint8_t* der_decode_boolean(CFAllocatorRef allocator,
                                  CFBooleanRef* boolean, CFErrorRef *error,
                                  const uint8_t* der, const uint8_t *der_end);

// CFData <-> DER
size_t der_sizeof_data(CFDataRef data, CFErrorRef *error);

uint8_t* der_encode_data(CFDataRef data, CFErrorRef *error,
                         const uint8_t *der, uint8_t *der_end);

const uint8_t* der_decode_data(CFAllocatorRef allocator,
                               CFDataRef* data, CFErrorRef *error,
                               const uint8_t* der, const uint8_t *der_end);

// CoreEntitlements -> CFData
// This is an opaque type
const uint8_t* der_decode_core_entitlements_data(CFAllocatorRef allocator,
                                                 CFDataRef* data, CFErrorRef *error,
                                                 const uint8_t* der, const uint8_t *der_end);

// CFDate <-> DER
size_t der_sizeof_date(CFDateRef date, CFErrorRef *error);

uint8_t* der_encode_date(CFDateRef date, CFErrorRef *error,
                         const uint8_t *der, uint8_t *der_end);

uint8_t* der_encode_date_repair(CFDateRef date, CFErrorRef *error,
                                bool repair, const uint8_t *der, uint8_t *der_end);

const uint8_t* der_decode_date(CFAllocatorRef allocator,
                               CFDateRef* date, CFErrorRef *error,
                               const uint8_t* der, const uint8_t *der_end);

const uint8_t* der_decode_utc_time(CFAllocatorRef allocator,
                                   CFDateRef* date, CFErrorRef *error,
                                   const uint8_t* der, const uint8_t *der_end);


// CFDictionary <-> DER
size_t der_sizeof_dictionary(CFDictionaryRef dictionary, CFErrorRef *error);

uint8_t* der_encode_dictionary(CFDictionaryRef dictionary, CFErrorRef *error,
                               const uint8_t *der, uint8_t *der_end);
uint8_t* der_encode_dictionary_repair(CFDictionaryRef dictionary, CFErrorRef *error,
                                      bool repair, const uint8_t *der, uint8_t *der_end);

const uint8_t* der_decode_dictionary(CFAllocatorRef allocator,
                                     CFDictionaryRef* dictionary, CFErrorRef *error,
                                     const uint8_t* der, const uint8_t *der_end);

// CFNumber <-> DER
// Currently only supports signed 64 bit values. No floating point.
size_t der_sizeof_number(CFNumberRef number, CFErrorRef *error);

uint8_t* der_encode_number(CFNumberRef number, CFErrorRef *error,
                           const uint8_t *der, uint8_t *der_end);

const uint8_t* der_decode_number(CFAllocatorRef allocator,
                                 CFNumberRef* number, CFErrorRef *error,
                                 const uint8_t* der, const uint8_t *der_end);

// CFString <-> DER
size_t der_sizeof_string(CFStringRef string, CFErrorRef *error);

uint8_t* der_encode_string(CFStringRef string, CFErrorRef *error,
                           const uint8_t *der, uint8_t *der_end);

const uint8_t* der_decode_string(CFAllocatorRef allocator,
                                 CFStringRef* string, CFErrorRef *error,
                                 const uint8_t* der, const uint8_t *der_end);

const uint8_t* der_decode_numeric_string(CFAllocatorRef allocator,
                                 CFStringRef* string, CFErrorRef *error,
                                 const uint8_t* der, const uint8_t *der_end);

// CFSet <-> DER
size_t der_sizeof_set(CFSetRef dict, CFErrorRef *error);

uint8_t* der_encode_set(CFSetRef set, CFErrorRef *error,
                        const uint8_t *der, uint8_t *der_end);
uint8_t* der_encode_set_repair(CFSetRef set, CFErrorRef *error,
                               bool repair, const uint8_t *der, uint8_t *der_end);

const uint8_t* der_decode_set(CFAllocatorRef allocator,
                              CFSetRef* set, CFErrorRef *error,
                              const uint8_t* der, const uint8_t *der_end);

#include <corecrypto/ccder.h>
enum {
    CCDER_CONSTRUCTED_CFSET = CCDER_PRIVATE | CCDER_SET,
};

#endif
