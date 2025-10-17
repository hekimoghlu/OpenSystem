/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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
/* $Id$ */

#ifndef __DER_H__
#define __DER_H__



typedef enum {
    ASN1_C_UNIV = 0,
    ASN1_C_APPL = 1,
    ASN1_C_CONTEXT = 2,
    ASN1_C_PRIVATE = 3
} Der_class;

typedef enum {PRIM = 0, CONS = 1} Der_type;

#define MAKE_TAG(CLASS, TYPE, TAG)  (((CLASS) << 6) | ((TYPE) << 5) | (TAG))

/* Universal tags */

enum {
    UT_EndOfContent	= 0,
    UT_Boolean		= 1,
    UT_Integer		= 2,
    UT_BitString	= 3,
    UT_OctetString	= 4,
    UT_Null		= 5,
    UT_OID		= 6,
    UT_Enumerated	= 10,
    UT_UTF8String	= 12,
    UT_Sequence		= 16,
    UT_Set		= 17,
    UT_PrintableString	= 19,
    UT_IA5String	= 22,
    UT_UTCTime		= 23,
    UT_GeneralizedTime	= 24,
    UT_UniversalString	= 25,
    UT_VisibleString	= 26,
    UT_GeneralString	= 27,
    UT_BMPString	= 30,
    /* unsupported types */
    UT_ObjectDescriptor = 7,
    UT_External		= 8,
    UT_Real		= 9,
    UT_EmbeddedPDV	= 11,
    UT_RelativeOID	= 13,
    UT_NumericString	= 18,
    UT_TeletexString	= 20,
    UT_VideotexString	= 21,
    UT_GraphicString	= 25
};

#define ASN1_INDEFINITE 0xdce0deed

#define ASN1_CHOICE_ELLIPSIS	(-1)
#define ASN1_CHOICE_INVALID	(0)

typedef struct heim_der_time_t {
    time_t dt_sec;
    unsigned long dt_nsec;
} heim_der_time_t;

typedef struct heim_ber_time_t {
    time_t bt_sec;
    unsigned bt_nsec;
    int bt_zone;
} heim_ber_time_t;

struct asn1_template;

#include <der-protos.h>

int _heim_fix_dce(size_t reallen, size_t *len);
int _heim_der_set_sort(const void *, const void *);
int _heim_time2generalizedtime (time_t, heim_octet_string *, int);

#endif /* __DER_H__ */
