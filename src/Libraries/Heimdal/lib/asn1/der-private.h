/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 10, 2025.
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
#ifndef __der_private_h__
#define __der_private_h__

#include <stdarg.h>

int
_asn1_bmember_isset_bit (
	const void */*data*/,
	unsigned int /*bit*/,
	size_t /*size*/);

void
_asn1_bmember_put_bit (
	unsigned char */*p*/,
	const void */*data*/,
	unsigned int /*bit*/,
	size_t /*size*/,
	unsigned int */*bitset*/);

void
_asn1_capture_data (
	const char */*type*/,
	const unsigned char */*p*/,
	size_t /*len*/);

int
_asn1_copy (
	const struct asn1_template */*t*/,
	const void */*from*/,
	void */*to*/);

int
_asn1_copy_top (
	const struct asn1_template */*t*/,
	const void */*from*/,
	void */*to*/);

int
_asn1_decode (
	const struct asn1_template */*t*/,
	unsigned /*flags*/,
	const unsigned char */*p*/,
	size_t /*len*/,
	void */*data*/,
	size_t */*size*/);

int
_asn1_decode_top (
	const struct asn1_template */*t*/,
	unsigned /*flags*/,
	const unsigned char */*p*/,
	size_t /*len*/,
	void */*data*/,
	size_t */*size*/);

int
_asn1_encode (
	const struct asn1_template */*t*/,
	unsigned char */*p*/,
	size_t /*len*/,
	const void */*data*/,
	size_t */*size*/);

int
_asn1_encode_fuzzer (
	const struct asn1_template */*t*/,
	unsigned char */*p*/,
	size_t /*len*/,
	const void */*data*/,
	size_t */*size*/);

void
_asn1_free (
	const struct asn1_template */*t*/,
	void */*data*/);

void
_asn1_free_top (
	const struct asn1_template */*t*/,
	void */*data*/);

size_t
_asn1_length (
	const struct asn1_template */*t*/,
	const void */*data*/);

size_t
_asn1_length_fuzzer (
	const struct asn1_template */*t*/,
	const void */*data*/);

size_t
_asn1_sizeofType (const struct asn1_template */*t*/);

int
_heim_der_set_sort (
	const void */*a1*/,
	const void */*a2*/);

int
_heim_fix_dce (
	size_t /*reallen*/,
	size_t */*len*/);

size_t
_heim_len_int (int /*val*/);

size_t
_heim_len_unsigned (unsigned /*val*/);

int
_heim_time2generalizedtime (
	time_t /*t*/,
	heim_octet_string */*s*/,
	int /*gtimep*/);

#endif /* __der_private_h__ */
