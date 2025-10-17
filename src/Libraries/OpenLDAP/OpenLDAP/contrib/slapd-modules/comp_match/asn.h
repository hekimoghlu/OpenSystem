/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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
/* ACKNOWLEDGEMENTS
 * This work originally developed by Sang Seok Lim
 * 2004/06/18	03:20:00	slim@OpenLDAP.org
 */
#ifndef _H_ASN_MODULE
#define _H_ASN_MODULE

typedef enum { BER, GSER } EncRulesType;

typedef enum AsnTypeId {
	BASICTYPE_BOOLEAN = 0,
	BASICTYPE_INTEGER, 
	BASICTYPE_BITSTRING,
	BASICTYPE_OCTETSTRING,
	BASICTYPE_NULL,
	BASICTYPE_OID,
	BASICTYPE_REAL,
	BASICTYPE_ENUMERATED,
	BASICTYPE_NUMERIC_STR,
	BASICTYPE_PRINTABLE_STR,
	BASICTYPE_UNIVERSAL_STR,
	BASICTYPE_IA5_STR,
	BASICTYPE_BMP_STR,
	BASICTYPE_UTF8_STR,
	BASICTYPE_UTCTIME,
	BASICTYPE_GENERALIZEDTIME,
	BASICTYPE_GRAPHIC_STR,
	BASICTYPE_VISIBLE_STR,
	BASICTYPE_GENERAL_STR,
	BASICTYPE_OBJECTDESCRIPTOR,
	BASICTYPE_VIDEOTEX_STR,
	BASICTYPE_T61_STR,
	BASICTYPE_OCTETCONTAINING,
	BASICTYPE_BITCONTAINING,
	BASICTYPE_RELATIVE_OID,	/* 25 */
	BASICTYPE_ANY,
	/* Embedded Composite Types*/
	COMPOSITE_ASN1_TYPE,
	/* A New ASN.1 types including type reference */
	RDNSequence,
	RelativeDistinguishedName,
	TelephoneNumber,
	FacsimileTelephoneNumber__telephoneNumber,
	DirectoryString,
	/* Newly Defined ASN.1 Type, Manually registered */
	ASN_COMP_CERTIFICATE,
	/* ASN.1 Type End */
	ASNTYPE_END
} AsnTypeId;

#endif
