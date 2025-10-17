/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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
#ifndef _CORECRYPTO_CCDER_BLOB_H_
#define _CORECRYPTO_CCDER_BLOB_H_

#include <corecrypto/cc.h>
#include <corecrypto/ccasn1.h>
#include <corecrypto/ccn.h>

#define CCDER_MULTIBYTE_TAGS 1

#ifdef CCDER_MULTIBYTE_TAGS
typedef unsigned long ccder_tag;
#else
typedef uint8_t ccder_tag;
#endif

typedef struct ccder_blob {
    uint8_t *cc_ended_by(der_end) der;
    uint8_t *der_end;
} ccder_blob;

typedef struct ccder_read_blob {
    const uint8_t *cc_ended_by(der_end) der;
    const uint8_t *der_end;
} ccder_read_blob;

#define ccder_size(BEGIN, END) ((size_t)((END) - (BEGIN)))
#define ccder_blob_size(BLOB) ccder_size((BLOB).der, (BLOB).der_end)

// MARK: - ccder_blob_encode_ functions.

CC_NONNULL((1)) CC_NODISCARD
bool ccder_blob_encode_tag(ccder_blob *into, ccder_tag tag);

CC_NONNULL((1)) CC_NODISCARD
bool ccder_blob_encode_len(ccder_blob *into, size_t len);

CC_NONNULL((1)) CC_NODISCARD
bool ccder_blob_encode_tl(ccder_blob *into, ccder_tag tag, size_t len);

CC_NONNULL((1)) CC_NODISCARD
bool ccder_blob_encode_body(ccder_blob *into, size_t size, const uint8_t *cc_sized_by(size) body);

CC_NONNULL((1, 4)) CC_NODISCARD
bool ccder_blob_encode_body_tl(ccder_blob *into, ccder_tag tag, size_t size, const uint8_t *cc_sized_by(size) body);

CC_NONNULL((1, 3)) CC_NODISCARD
bool ccder_blob_reserve(ccder_blob *into, size_t reserve_size, ccder_blob *out_reserved);

CC_NONNULL((1, 4)) CC_NODISCARD
bool ccder_blob_reserve_tl(ccder_blob *into, ccder_tag tag, size_t reserve_size, ccder_blob *out_reserved);

CC_NONNULL((1, 2)) CC_NODISCARD
bool ccder_blob_encode_oid(ccder_blob *into, ccoid_t oid);

CC_NONNULL((1, 4)) CC_NODISCARD
bool ccder_blob_encode_implicit_integer(ccder_blob *into, ccder_tag implicit_tag, cc_size n, const cc_unit *cc_counted_by(n) s);

CC_NONNULL((1, 3)) CC_NODISCARD
bool ccder_blob_encode_integer(ccder_blob *into, cc_size n, const cc_unit *cc_counted_by(n) s);

CC_NONNULL((1)) CC_NODISCARD
bool ccder_blob_encode_implicit_uint64(ccder_blob *into, ccder_tag implicit_tag, uint64_t value);

CC_NONNULL((1)) CC_NODISCARD
bool ccder_blob_encode_uint64(ccder_blob *into, uint64_t value);

CC_NONNULL((1, 3)) CC_NODISCARD
bool ccder_blob_encode_octet_string(ccder_blob *into, cc_size n, const cc_unit *cc_counted_by(n) s);

CC_NONNULL((1, 4)) CC_NODISCARD
bool ccder_blob_encode_implicit_octet_string(ccder_blob *into, ccder_tag implicit_tag, cc_size n, const cc_unit *cc_counted_by(n) s);

CC_NONNULL((1, 4)) CC_NODISCARD
bool ccder_blob_encode_implicit_raw_octet_string(ccder_blob *into, ccder_tag implicit_tag, size_t s_size, const uint8_t *cc_sized_by(s_size) s);

CC_NONNULL((1, 3)) CC_NODISCARD
bool ccder_blob_encode_raw_octet_string(ccder_blob *into, size_t s_size, const uint8_t *cc_sized_by(s_size) s);

CC_NONNULL((1, 3)) CC_NODISCARD
bool ccder_blob_encode_eckey(ccder_blob *into, size_t priv_byte_size, const uint8_t *cc_sized_by(priv_byte_size) priv_key, ccoid_t oid, size_t pub_byte_size, const uint8_t *cc_sized_by(pub_byte_size) pub_key);

// MARK: - ccder_blob_decode_ functions.
CC_NONNULL((1, 2)) CC_NODISCARD
bool ccder_blob_decode_tag(ccder_read_blob *from, ccder_tag *tag);

CC_NONNULL((1, 2)) CC_NODISCARD
bool ccder_blob_decode_len(ccder_read_blob *from, size_t *size);

CC_NONNULL((1, 2)) CC_NODISCARD
bool ccder_blob_decode_len_strict(ccder_read_blob *from, size_t *size);

CC_NONNULL((1, 3)) CC_NODISCARD
bool ccder_blob_decode_tl(ccder_read_blob *from, ccder_tag expected_tag, size_t *size);

CC_NONNULL((1, 3)) CC_NODISCARD
bool ccder_blob_decode_tl_strict(ccder_read_blob *from, ccder_tag expected_tag, size_t *size);

CC_NONNULL((1, 3)) CC_NODISCARD
bool ccder_blob_decode_range(ccder_read_blob *from, ccder_tag expected_tag, ccder_read_blob *range_blob);

CC_NONNULL((1, 3)) CC_NODISCARD
bool ccder_blob_decode_range_strict(ccder_read_blob *from, ccder_tag expected_tag, ccder_read_blob *range_blob);

CC_NONNULL((1, 2)) CC_NODISCARD
bool ccder_blob_decode_sequence_tl(ccder_read_blob *from, ccder_read_blob *range_blob);

CC_NONNULL((1, 2)) CC_NODISCARD
bool ccder_blob_decode_sequence_tl_strict(ccder_read_blob *from, ccder_read_blob *range_blob);

CC_NONNULL((1, 2)) CC_NODISCARD
bool ccder_blob_decode_uint_n(ccder_read_blob *from, cc_size *n);

CC_NONNULL((1)) CC_NODISCARD
bool ccder_blob_decode_uint64(ccder_read_blob *from, uint64_t *r);

CC_NONNULL((1, 3)) CC_NODISCARD
bool ccder_blob_decode_uint(ccder_read_blob *from, cc_size n, cc_unit *cc_counted_by(n));

CC_NONNULL((1, 3)) CC_NODISCARD
bool ccder_blob_decode_uint_strict(ccder_read_blob *from, cc_size n, cc_unit *cc_counted_by(n));

CC_NONNULL((1, 3, 4)) CC_NODISCARD
bool ccder_blob_decode_seqii(ccder_read_blob *from, size_t n, cc_unit *cc_counted_by(n) r, cc_unit *cc_counted_by(n) s);

CC_NONNULL((1, 3, 4)) CC_NODISCARD
bool ccder_blob_decode_seqii_strict(ccder_read_blob *from, size_t n, cc_unit *cc_counted_by(n) r, cc_unit *cc_counted_by(n) s);

CC_NONNULL((1, 2)) CC_NODISCARD
bool ccder_blob_decode_oid(ccder_read_blob *from, ccoid_t *oidp);

CC_NONNULL((1, 2, 3)) CC_NODISCARD
bool ccder_blob_decode_bitstring(ccder_read_blob *from, ccder_read_blob *bit_string_range, size_t *bit_count);

CC_NONNULL((1, 2, 3, 4, 5, 6, 7)) CC_NODISCARD
bool ccder_blob_decode_eckey(ccder_read_blob *from, uint64_t *version, size_t *priv_key_byte_size, const uint8_t *cc_sized_by(*priv_key_byte_size) *priv_key, ccoid_t *oid, size_t *pub_key_byte_size, const uint8_t *cc_sized_by(*pub_key_byte_size) *pub_key, size_t *pub_key_bit_count);

#endif /* _CORECRYPTO_CCDER_BLOB_H_ */
