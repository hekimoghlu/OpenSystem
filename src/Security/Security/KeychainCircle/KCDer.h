/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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
//  KCDer.h
//  KeychainCircle
//
//

#include <Foundation/Foundation.h>
#include <corecrypto/ccder.h>

NS_ASSUME_NONNULL_BEGIN

// These should probably be shared with security, but we don't export our der'izing functions yet.
const uint8_t* _Nullable
kcder_decode_data_nocopy(NSData* _Nullable * _Nonnull data,
                         NSError* _Nullable * _Nullable error,
                         const uint8_t* _Nonnull der, const uint8_t * _Nullable der_end);
const uint8_t* _Nullable
kcder_decode_data(NSData* _Nullable* _Nonnull data, NSError* _Nullable * _Nullable error,
                  const uint8_t* der, const uint8_t * _Nullable der_end);
size_t
kcder_sizeof_data(NSData* data, NSError** error);
uint8_t* _Nullable
kcder_encode_data(NSData* data, NSError**error,
                  const uint8_t * _Nonnull der, uint8_t * _Nullable der_end);
uint8_t* _Nullable
kcder_encode_data_optional(NSData* _Nullable data, NSError* _Nullable * _Nullable error,
                           const uint8_t *der, uint8_t *der_end);

const uint8_t* _Nullable
kcder_decode_string(NSString*_Nullable * _Nonnull string,
                    NSError* _Nullable * _Nullable error,
                    const uint8_t* _Nonnull der,
                    const uint8_t* _Nullable der_end);
size_t
kcder_sizeof_string(NSString* string,
                    NSError* _Nullable * _Nullable error);
uint8_t* _Nullable
kcder_encode_string(NSString* string,
                    NSError* _Nullable * _Nullable error,
                    const uint8_t * _Nonnull der, uint8_t * _Nullable der_end);

uint8_t *
kcder_encode_raw_octet_space(size_t s_size, uint8_t * _Nullable * _Nonnull location,
                                      const uint8_t * _Nonnull der, uint8_t * _Nullable der_end);

NS_ASSUME_NONNULL_END
