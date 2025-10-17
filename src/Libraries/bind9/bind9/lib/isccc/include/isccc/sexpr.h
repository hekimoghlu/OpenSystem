/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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
/* $Id: sexpr.h,v 1.11 2007/08/28 07:20:43 tbox Exp $ */

#ifndef ISCCC_SEXPR_H
#define ISCCC_SEXPR_H 1

/*! \file isccc/sexpr.h */

#include <stdio.h>

#include <isc/lang.h>
#include <isccc/types.h>

ISC_LANG_BEGINDECLS

/*% dotted pair structure */
struct isccc_dottedpair {
	isccc_sexpr_t *car;
	isccc_sexpr_t *cdr;
};

/*% iscc_sexpr structure */
struct isccc_sexpr {
	unsigned int			type;
	union {
		char *			as_string;
		isccc_dottedpair_t	as_dottedpair;
		isccc_region_t		as_region;
	}				value;
};

#define ISCCC_SEXPRTYPE_NONE		0x00	/*%< Illegal. */
#define ISCCC_SEXPRTYPE_T			0x01
#define ISCCC_SEXPRTYPE_STRING		0x02
#define ISCCC_SEXPRTYPE_DOTTEDPAIR	0x03
#define ISCCC_SEXPRTYPE_BINARY		0x04

#define ISCCC_SEXPR_CAR(s)		(s)->value.as_dottedpair.car
#define ISCCC_SEXPR_CDR(s)		(s)->value.as_dottedpair.cdr

isccc_sexpr_t *
isccc_sexpr_cons(isccc_sexpr_t *car, isccc_sexpr_t *cdr);

isccc_sexpr_t *
isccc_sexpr_tconst(void);

isccc_sexpr_t *
isccc_sexpr_fromstring(const char *str);

isccc_sexpr_t *
isccc_sexpr_frombinary(const isccc_region_t *region);

void
isccc_sexpr_free(isccc_sexpr_t **sexprp);

void
isccc_sexpr_print(isccc_sexpr_t *sexpr, FILE *stream);

isccc_sexpr_t *
isccc_sexpr_car(isccc_sexpr_t *list);

isccc_sexpr_t *
isccc_sexpr_cdr(isccc_sexpr_t *list);

void
isccc_sexpr_setcar(isccc_sexpr_t *pair, isccc_sexpr_t *car);

void
isccc_sexpr_setcdr(isccc_sexpr_t *pair, isccc_sexpr_t *cdr);

isccc_sexpr_t *
isccc_sexpr_addtolist(isccc_sexpr_t **l1p, isccc_sexpr_t *l2);

isc_boolean_t
isccc_sexpr_listp(isccc_sexpr_t *sexpr);

isc_boolean_t
isccc_sexpr_emptyp(isccc_sexpr_t *sexpr);

isc_boolean_t
isccc_sexpr_stringp(isccc_sexpr_t *sexpr);

isc_boolean_t
isccc_sexpr_binaryp(isccc_sexpr_t *sexpr);

char *
isccc_sexpr_tostring(isccc_sexpr_t *sexpr);

isccc_region_t *
isccc_sexpr_tobinary(isccc_sexpr_t *sexpr);

ISC_LANG_ENDDECLS

#endif /* ISCCC_SEXPR_H */
