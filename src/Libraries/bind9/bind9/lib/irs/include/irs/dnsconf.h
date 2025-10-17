/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 29, 2025.
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
/* $Id: dnsconf.h,v 1.3 2009/09/02 23:48:02 tbox Exp $ */

#ifndef IRS_DNSCONF_H
#define IRS_DNSCONF_H 1

/*! \file
 *
 * \brief
 * The IRS dnsconf module parses an "advanced" configuration file related to
 * the DNS library, such as trusted keys for DNSSEC validation, and creates
 * the corresponding configuration objects for the DNS library modules.
 *
 * Notes:
 * This module is very experimental and the configuration syntax or library
 * interfaces may change in future versions.  Currently, only the
 * 'trusted-keys' statement is supported, whose syntax is the same as the
 * same name of statement for named.conf.
 */

#include <irs/types.h>

/*%
 * A compound structure storing DNS key information mainly for DNSSEC
 * validation.  A dns_key_t object will be created using the 'keyname' and
 * 'keydatabuf' members with the dst_key_fromdns() function.
 */
typedef struct irs_dnsconf_dnskey {
	dns_name_t				*keyname;
	isc_buffer_t				*keydatabuf;
	ISC_LINK(struct irs_dnsconf_dnskey)	link;
} irs_dnsconf_dnskey_t;

typedef ISC_LIST(irs_dnsconf_dnskey_t) irs_dnsconf_dnskeylist_t;

ISC_LANG_BEGINDECLS

isc_result_t
irs_dnsconf_load(isc_mem_t *mctx, const char *filename, irs_dnsconf_t **confp);
/*%<
 * Load the "advanced" DNS configuration file 'filename' in the "dns.conf"
 * format, and create a new irs_dnsconf_t object from the configuration.
 *
 * Requires:
 *
 *\li	'mctx' is a valid memory context.
 *
 *\li	'filename' != NULL
 *
 *\li	'confp' != NULL && '*confp' == NULL
 */

void
irs_dnsconf_destroy(irs_dnsconf_t **confp);
/*%<
 * Destroy the dnsconf object.
 *
 * Requires:
 *
 *\li	'*confp' is a valid dnsconf object.
 *
 * Ensures:
 *
 *\li	*confp == NULL
 */

irs_dnsconf_dnskeylist_t *
irs_dnsconf_gettrustedkeys(irs_dnsconf_t *conf);
/*%<
 * Return a list of key information stored in 'conf'.
 *
 * Requires:
 *
 *\li	'conf' is a valid dnsconf object.
 */

ISC_LANG_ENDDECLS

#endif /* IRS_DNSCONF_H */
