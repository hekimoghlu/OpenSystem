/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
/* $Id: dnsconf.c,v 1.4 2009/09/02 23:48:03 tbox Exp $ */

/*! \file */

#include <config.h>

#include <isccfg/cfg.h>
#include <isccfg/grammar.h>

/*%
 * A trusted key, as used in the "trusted-keys" statement.
 */
static cfg_tuplefielddef_t trustedkey_fields[] = {
	{ "name", &cfg_type_astring, 0 },
	{ "flags", &cfg_type_uint32, 0 },
	{ "protocol", &cfg_type_uint32, 0 },
	{ "algorithm", &cfg_type_uint32, 0 },
	{ "key", &cfg_type_qstring, 0 },
	{ NULL, NULL, 0 }
};

static cfg_type_t cfg_type_trustedkey = {
	"trustedkey", cfg_parse_tuple, cfg_print_tuple, cfg_doc_tuple,
	&cfg_rep_tuple, trustedkey_fields
};

static cfg_type_t cfg_type_trustedkeys = {
	"trusted-keys", cfg_parse_bracketed_list, cfg_print_bracketed_list,
	cfg_doc_bracketed_list, &cfg_rep_list, &cfg_type_trustedkey
};

/*%
 * Clauses that can be found within the top level of the dns.conf
 * file only.
 */
static cfg_clausedef_t
dnsconf_clauses[] = {
	{ "trusted-keys", &cfg_type_trustedkeys, CFG_CLAUSEFLAG_MULTI },
	{ NULL, NULL, 0 }
};

/*% The top-level dns.conf syntax. */

static cfg_clausedef_t *
dnsconf_clausesets[] = {
	dnsconf_clauses,
	NULL
};

LIBISCCFG_EXTERNAL_DATA cfg_type_t cfg_type_dnsconf = {
	"dnsconf", cfg_parse_mapbody, cfg_print_mapbody, cfg_doc_mapbody,
	&cfg_rep_map, dnsconf_clausesets
};
