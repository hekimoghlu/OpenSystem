/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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
/* $Id: dst_parse.h,v 1.17 2010/12/23 23:47:08 tbox Exp $ */

/*! \file */
#ifndef DST_DST_PARSE_H
#define DST_DST_PARSE_H 1

#include <isc/lang.h>

#include <dst/dst.h>

#define MAXFIELDSIZE		512

/*
 * Maximum number of fields in a private file is 18 (12 algorithm-
 * specific fields for RSA, plus 6 generic fields).
 */
#define MAXFIELDS		12+6

#define TAG_SHIFT		4
#define TAG_ALG(tag)		((unsigned int)(tag) >> TAG_SHIFT)
#define TAG(alg, off)		(((alg) << TAG_SHIFT) + (off))

/* These are used by both RSA-MD5 and RSA-SHA1 */
#define RSA_NTAGS		11
#define TAG_RSA_MODULUS		((DST_ALG_RSAMD5 << TAG_SHIFT) + 0)
#define TAG_RSA_PUBLICEXPONENT	((DST_ALG_RSAMD5 << TAG_SHIFT) + 1)
#define TAG_RSA_PRIVATEEXPONENT	((DST_ALG_RSAMD5 << TAG_SHIFT) + 2)
#define TAG_RSA_PRIME1		((DST_ALG_RSAMD5 << TAG_SHIFT) + 3)
#define TAG_RSA_PRIME2		((DST_ALG_RSAMD5 << TAG_SHIFT) + 4)
#define TAG_RSA_EXPONENT1	((DST_ALG_RSAMD5 << TAG_SHIFT) + 5)
#define TAG_RSA_EXPONENT2	((DST_ALG_RSAMD5 << TAG_SHIFT) + 6)
#define TAG_RSA_COEFFICIENT	((DST_ALG_RSAMD5 << TAG_SHIFT) + 7)
#define TAG_RSA_ENGINE		((DST_ALG_RSAMD5 << TAG_SHIFT) + 8)
#define TAG_RSA_LABEL		((DST_ALG_RSAMD5 << TAG_SHIFT) + 9)

#define DH_NTAGS		4
#define TAG_DH_PRIME		((DST_ALG_DH << TAG_SHIFT) + 0)
#define TAG_DH_GENERATOR	((DST_ALG_DH << TAG_SHIFT) + 1)
#define TAG_DH_PRIVATE		((DST_ALG_DH << TAG_SHIFT) + 2)
#define TAG_DH_PUBLIC		((DST_ALG_DH << TAG_SHIFT) + 3)

#define DSA_NTAGS		5
#define TAG_DSA_PRIME		((DST_ALG_DSA << TAG_SHIFT) + 0)
#define TAG_DSA_SUBPRIME	((DST_ALG_DSA << TAG_SHIFT) + 1)
#define TAG_DSA_BASE		((DST_ALG_DSA << TAG_SHIFT) + 2)
#define TAG_DSA_PRIVATE		((DST_ALG_DSA << TAG_SHIFT) + 3)
#define TAG_DSA_PUBLIC		((DST_ALG_DSA << TAG_SHIFT) + 4)

#define GOST_NTAGS		1
#define TAG_GOST_PRIVASN1	((DST_ALG_ECCGOST << TAG_SHIFT) + 0)
#define TAG_GOST_PRIVRAW	((DST_ALG_ECCGOST << TAG_SHIFT) + 1)

#define ECDSA_NTAGS		4
#define TAG_ECDSA_PRIVATEKEY	((DST_ALG_ECDSA256 << TAG_SHIFT) + 0)
#define TAG_ECDSA_ENGINE	((DST_ALG_ECDSA256 << TAG_SHIFT) + 1)
#define TAG_ECDSA_LABEL		((DST_ALG_ECDSA256 << TAG_SHIFT) + 2)

#define OLD_HMACMD5_NTAGS	1
#define HMACMD5_NTAGS		2
#define TAG_HMACMD5_KEY		((DST_ALG_HMACMD5 << TAG_SHIFT) + 0)
#define TAG_HMACMD5_BITS	((DST_ALG_HMACMD5 << TAG_SHIFT) + 1)

#define HMACSHA1_NTAGS		2
#define TAG_HMACSHA1_KEY	((DST_ALG_HMACSHA1 << TAG_SHIFT) + 0)
#define TAG_HMACSHA1_BITS	((DST_ALG_HMACSHA1 << TAG_SHIFT) + 1)

#define HMACSHA224_NTAGS	2
#define TAG_HMACSHA224_KEY	((DST_ALG_HMACSHA224 << TAG_SHIFT) + 0)
#define TAG_HMACSHA224_BITS	((DST_ALG_HMACSHA224 << TAG_SHIFT) + 1)

#define HMACSHA256_NTAGS	2
#define TAG_HMACSHA256_KEY	((DST_ALG_HMACSHA256 << TAG_SHIFT) + 0)
#define TAG_HMACSHA256_BITS	((DST_ALG_HMACSHA256 << TAG_SHIFT) + 1)

#define HMACSHA384_NTAGS	2
#define TAG_HMACSHA384_KEY	((DST_ALG_HMACSHA384 << TAG_SHIFT) + 0)
#define TAG_HMACSHA384_BITS	((DST_ALG_HMACSHA384 << TAG_SHIFT) + 1)

#define HMACSHA512_NTAGS	2
#define TAG_HMACSHA512_KEY	((DST_ALG_HMACSHA512 << TAG_SHIFT) + 0)
#define TAG_HMACSHA512_BITS	((DST_ALG_HMACSHA512 << TAG_SHIFT) + 1)

struct dst_private_element {
	unsigned short tag;
	unsigned short length;
	unsigned char *data;
};

typedef struct dst_private_element dst_private_element_t;

struct dst_private {
	unsigned short nelements;
	dst_private_element_t elements[MAXFIELDS];
};

typedef struct dst_private dst_private_t;

ISC_LANG_BEGINDECLS

void
dst__privstruct_free(dst_private_t *priv, isc_mem_t *mctx);

isc_result_t
dst__privstruct_parse(dst_key_t *key, unsigned int alg, isc_lex_t *lex,
		      isc_mem_t *mctx, dst_private_t *priv);

isc_result_t
dst__privstruct_writefile(const dst_key_t *key, const dst_private_t *priv,
			  const char *directory);

ISC_LANG_ENDDECLS

#endif /* DST_DST_PARSE_H */
