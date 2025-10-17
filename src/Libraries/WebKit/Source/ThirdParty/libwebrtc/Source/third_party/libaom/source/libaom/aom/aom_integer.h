/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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
#ifndef AOM_AOM_AOM_INTEGER_H_
#define AOM_AOM_AOM_INTEGER_H_

/* get ptrdiff_t, size_t, wchar_t, NULL */
#include <stddef.h>  // IWYU pragma: export

/* Assume platforms have the C99 standard integer types. */

#if defined(__cplusplus)
#if !defined(__STDC_FORMAT_MACROS)
#define __STDC_FORMAT_MACROS
#endif
#if !defined(__STDC_LIMIT_MACROS)
#define __STDC_LIMIT_MACROS
#endif
#endif  // __cplusplus

#include <stdint.h>    // IWYU pragma: export
#include <inttypes.h>  // IWYU pragma: export

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

// Returns size of uint64_t when encoded using LEB128.
size_t aom_uleb_size_in_bytes(uint64_t value);

// Returns 0 on success, -1 on decode failure.
// On success, 'value' stores the decoded LEB128 value and 'length' stores
// the number of bytes decoded.
int aom_uleb_decode(const uint8_t *buffer, size_t available, uint64_t *value,
                    size_t *length);

// Encodes LEB128 integer. Returns 0 when successful, and -1 upon failure.
int aom_uleb_encode(uint64_t value, size_t available, uint8_t *coded_value,
                    size_t *coded_size);

// Encodes LEB128 integer to size specified. Returns 0 when successful, and -1
// upon failure.
// Note: This will write exactly pad_to_size bytes; if the value cannot be
// encoded in this many bytes, then this will fail.
int aom_uleb_encode_fixed_size(uint64_t value, size_t available,
                               size_t pad_to_size, uint8_t *coded_value,
                               size_t *coded_size);

#if defined(__cplusplus)
}  // extern "C"
#endif  // __cplusplus

#endif  // AOM_AOM_AOM_INTEGER_H_
