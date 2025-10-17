/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 30, 2024.
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
/* $Id: keygen.h,v 1.3 2009/06/11 23:47:55 tbox Exp $ */

#ifndef RNDC_KEYGEN_H
#define RNDC_KEYGEN_H 1

/*! \file */

#include <isc/lang.h>

ISC_LANG_BEGINDECLS

void generate_key(isc_mem_t *mctx, const char *randomfile, dns_secalg_t alg,
		  int keysize, isc_buffer_t *key_txtbuffer);

void write_key_file(const char *keyfile, const char *user,
		    const char *keyname, isc_buffer_t *secret,
		    dns_secalg_t alg);

const char *alg_totext(dns_secalg_t alg);
dns_secalg_t alg_fromtext(const char *name);
int alg_bits(dns_secalg_t alg);

ISC_LANG_ENDDECLS

#endif /* RNDC_KEYGEN_H */
