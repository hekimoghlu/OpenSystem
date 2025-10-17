/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
/* asn1 templates */

#ifndef __TEMPLATE_H__
#define __TEMPLATE_H__

#ifndef HEIMDAL_PRINTF_ATTRIBUTE
#if defined(__GNUC__) && ((__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 1 )))
#define HEIMDAL_PRINTF_ATTRIBUTE(x) __attribute__((format x))
#else
#define HEIMDAL_PRINTF_ATTRIBUTE(x)
#endif
#endif

/* tag:
 *  0..20 tag
 * 21     type
 * 22..23 class
 * 24..27 flags
 * 28..31 op
 */

/* parse:
 *  0..11 type
 * 12..23 unused
 * 24..27 flags
 * 28..31 op
 */

#define A1_OP_MASK		(0xf0000000)
#define A1_OP_TYPE		(0x10000000)
#define A1_OP_TYPE_EXTERN	(0x20000000)
#define A1_OP_TAG		(0x30000000)
#define A1_OP_PARSE		(0x40000000)
#define A1_OP_SEQOF		(0x50000000)
#define A1_OP_SETOF		(0x60000000)
#define A1_OP_BMEMBER		(0x70000000)
#define A1_OP_CHOICE		(0x80000000)

#define A1_FLAG_MASK		(0x0f000000)
#define A1_FLAG_OPTIONAL	(0x01000000)
#define A1_FLAG_IMPLICIT	(0x02000000)

#define A1_TAG_T(CLASS,TYPE,TAG)	((A1_OP_TAG) | (((CLASS) << 22) | ((TYPE) << 21) | (TAG)))
#define A1_TAG_CLASS(x)		(((x) >> 22) & 0x3)
#define A1_TAG_TYPE(x)		(((x) >> 21) & 0x1)
#define A1_TAG_TAG(x)		((x) & 0x1fffff)

#define A1_TAG_LEN(t)		((uintptr_t)(t)->ptr)
#define A1_HEADER_LEN(t)	((uintptr_t)(t)->ptr)

#define A1_PARSE_T(type)	((A1_OP_PARSE) | (type))
#define A1_PARSE_TYPE_MASK	0xfff
#define A1_PARSE_TYPE(x)	(A1_PARSE_TYPE_MASK & (x))

#define A1_PF_NESTED_INDEFINITE	0x1
#define A1_PF_ALLOW_BER		0x2
#define A1_PF_INDEFINITE	0x4

#define A1_HF_PRESERVE		0x1
#define A1_HF_ELLIPSIS		0x2

#define A1_HBF_RFC1510		0x1


struct asn1_template {
    uint32_t tt;
    uint32_t offset;
    const void *ptr;
};

typedef int (*asn1_type_decode)(const unsigned char *, size_t, void *, size_t *);
typedef int (*asn1_type_encode)(unsigned char *, size_t, const void *, size_t *);
typedef size_t (*asn1_type_length)(const void *);
typedef void (*asn1_type_release)(void *);
typedef int (*asn1_type_copy)(const void *, void *);

struct asn1_type_func {
    asn1_type_encode encode;
    asn1_type_decode decode;
    asn1_type_length length;
    asn1_type_copy copy;
    asn1_type_release release;
    size_t size;
};

struct template_of {
    unsigned int len;
    void *val;
};

enum template_types {
    A1T_IMEMBER = 0,
    A1T_HEIM_INTEGER,
    A1T_INTEGER,
    A1T_UNSIGNED,
    A1T_GENERAL_STRING,
    A1T_OCTET_STRING,
    A1T_OCTET_STRING_BER,
    A1T_IA5_STRING,
    A1T_BMP_STRING,
    A1T_UNIVERSAL_STRING,
    A1T_PRINTABLE_STRING,
    A1T_VISIBLE_STRING,
    A1T_UTF8_STRING,
    A1T_GENERALIZED_TIME,
    A1T_UTC_TIME,
    A1T_HEIM_BIT_STRING,
    A1T_BOOLEAN,
    A1T_OID,
    A1T_TELETEX_STRING,
    A1T_NUM_ENTRY
};

extern struct asn1_type_func asn1_template_prim[A1T_NUM_ENTRY];

#define ABORT_ON_ERROR(...) asn1_abort(__VA_ARGS__)

#define DPOC(data,offset) ((const void *)(((const unsigned char *)data)  + offset))
#define DPO(data,offset) ((void *)(((unsigned char *)data)  + offset))

/*
 * These functions are needed by the generated template stubs and are
 * really internal functions. Since they are part of der-private.h
 * that contains extra prototypes that really a private we included a
 * copy here.
 */

int
_asn1_copy_top (
	const struct asn1_template */*t*/,
	const void */*from*/,
	void */*to*/);

void
_asn1_free_top(const struct asn1_template *t,
	       void *data);

void
_asn1_capture_data(const char *type, const unsigned char *p, size_t len);

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

size_t
_asn1_length (
	const struct asn1_template */*t*/,
	const void */*data*/);

size_t
_asn1_length_fuzzer (
	const struct asn1_template */*t*/,
	const void */*data*/);


#endif
