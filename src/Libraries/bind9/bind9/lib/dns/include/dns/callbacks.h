/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
/* $Id: callbacks.h,v 1.26.40.1 2012/02/07 00:44:16 each Exp $ */

#ifndef DNS_CALLBACKS_H
#define DNS_CALLBACKS_H 1

/*! \file dns/callbacks.h */

/***
 ***	Imports
 ***/

#include <isc/lang.h>
#include <isc/magic.h>

#include <dns/types.h>

ISC_LANG_BEGINDECLS

/***
 ***	Types
 ***/

#define DNS_CALLBACK_MAGIC	ISC_MAGIC('C','L','L','B')
#define DNS_CALLBACK_VALID(cb)	ISC_MAGIC_VALID(cb, DNS_CALLBACK_MAGIC)

struct dns_rdatacallbacks {
	unsigned int magic;

	/*%
	 * dns_load_master calls this when it has rdatasets to commit.
	 */
	dns_addrdatasetfunc_t add;

	/*%
	 * This is called when reading in a database image from a 'map'
	 * format zone file.
	 */
	dns_deserializefunc_t deserialize;

	/*%
	 * dns_master_load*() call this when loading a raw zonefile,
	 * to pass back information obtained from the file header
	 */
	dns_rawdatafunc_t rawdata;
	dns_zone_t *zone;

	/*%
	 * dns_load_master / dns_rdata_fromtext call this to issue a error.
	 */
	void	(*error)(struct dns_rdatacallbacks *, const char *, ...);
	/*%
	 * dns_load_master / dns_rdata_fromtext call this to issue a warning.
	 */
	void	(*warn)(struct dns_rdatacallbacks *, const char *, ...);
	/*%
	 * Private data handles for use by the above callback functions.
	 */
	void	*add_private;
	void	*deserialize_private;
	void	*error_private;
	void	*warn_private;
};

/***
 ***	Initialization
 ***/

void
dns_rdatacallbacks_init(dns_rdatacallbacks_t *callbacks);
/*%<
 * Initialize 'callbacks'.
 *
 * \li	'magic' is set to DNS_CALLBACK_MAGIC
 *
 * \li	'error' and 'warn' are set to default callbacks that print the
 *	error message through the DNS library log context.
 *
 *\li	All other elements are initialized to NULL.
 *
 * Requires:
 *  \li    'callbacks' is a valid dns_rdatacallbacks_t,
 */

void
dns_rdatacallbacks_init_stdio(dns_rdatacallbacks_t *callbacks);
/*%<
 * Like dns_rdatacallbacks_init, but logs to stdio.
 */

ISC_LANG_ENDDECLS

#endif /* DNS_CALLBACKS_H */
