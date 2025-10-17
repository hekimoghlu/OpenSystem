/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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
/* $Id: alist.h,v 1.10 2007/08/28 07:20:43 tbox Exp $ */

#ifndef ISCCC_ALIST_H
#define ISCCC_ALIST_H 1

/*! \file isccc/alist.h */

#include <stdio.h>

#include <isc/lang.h>
#include <isccc/types.h>

ISC_LANG_BEGINDECLS

isccc_sexpr_t *
isccc_alist_create(void);

isc_boolean_t
isccc_alist_alistp(isccc_sexpr_t *alist);

isc_boolean_t
isccc_alist_emptyp(isccc_sexpr_t *alist);

isccc_sexpr_t *
isccc_alist_first(isccc_sexpr_t *alist);

isccc_sexpr_t *
isccc_alist_assq(isccc_sexpr_t *alist, const char *key);

void
isccc_alist_delete(isccc_sexpr_t *alist, const char *key);

isccc_sexpr_t *
isccc_alist_define(isccc_sexpr_t *alist, const char *key, isccc_sexpr_t *value);

isccc_sexpr_t *
isccc_alist_definestring(isccc_sexpr_t *alist, const char *key, const char *str);

isccc_sexpr_t *
isccc_alist_definebinary(isccc_sexpr_t *alist, const char *key, isccc_region_t *r);

isccc_sexpr_t *
isccc_alist_lookup(isccc_sexpr_t *alist, const char *key);

isc_result_t
isccc_alist_lookupstring(isccc_sexpr_t *alist, const char *key, char **strp);

isc_result_t
isccc_alist_lookupbinary(isccc_sexpr_t *alist, const char *key, isccc_region_t **r);

void
isccc_alist_prettyprint(isccc_sexpr_t *sexpr, unsigned int indent, FILE *stream);

ISC_LANG_ENDDECLS

#endif /* ISCCC_ALIST_H */
