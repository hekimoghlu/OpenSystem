/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
/* $Id: rdatalist.h,v 1.22 2008/04/03 06:09:05 tbox Exp $ */

#ifndef DNS_RDATALIST_H
#define DNS_RDATALIST_H 1

/*****
 ***** Module Info
 *****/

/*! \file dns/rdatalist.h
 * \brief
 * A DNS rdatalist is a list of rdata of a common type and class.
 *
 * MP:
 *\li	Clients of this module must impose any required synchronization.
 *
 * Reliability:
 *\li	No anticipated impact.
 *
 * Resources:
 *\li	TBS
 *
 * Security:
 *\li	No anticipated impact.
 *
 * Standards:
 *\li	None.
 */

#include <isc/lang.h>

#include <dns/types.h>

/*%
 * Clients may use this type directly.
 */
struct dns_rdatalist {
	dns_rdataclass_t		rdclass;
	dns_rdatatype_t			type;
	dns_rdatatype_t			covers;
	dns_ttl_t			ttl;
	ISC_LIST(dns_rdata_t)		rdata;
	ISC_LINK(dns_rdatalist_t)	link;
};

ISC_LANG_BEGINDECLS

void
dns_rdatalist_init(dns_rdatalist_t *rdatalist);
/*%<
 * Initialize rdatalist.
 *
 * Ensures:
 *\li	All fields of rdatalist have been initialized to their default
 *	values.
 */

isc_result_t
dns_rdatalist_tordataset(dns_rdatalist_t *rdatalist,
			 dns_rdataset_t *rdataset);
/*%<
 * Make 'rdataset' refer to the rdata in 'rdatalist'.
 *
 * Note:
 *\li	The caller must ensure that 'rdatalist' remains valid and unchanged
 *	while 'rdataset' is associated with it.
 *
 * Requires:
 *
 *\li	'rdatalist' is a valid rdatalist.
 *
 *\li	'rdataset' is a valid rdataset that is not currently associated with
 *	any rdata.
 *
 * Ensures,
 *	on success,
 *
 *\li		'rdataset' is associated with the rdata in rdatalist.
 *
 * Returns:
 *\li	#ISC_R_SUCCESS
 */

isc_result_t
dns_rdatalist_fromrdataset(dns_rdataset_t *rdataset,
			   dns_rdatalist_t **rdatalist);
/*%<
 * Point 'rdatalist' to the rdatalist in 'rdataset'.
 *
 * Requires:
 *
 *\li	'rdatalist' is a pointer to a NULL dns_rdatalist_t pointer.
 *
 *\li	'rdataset' is a valid rdataset associated with an rdatalist.
 *
 * Ensures,
 *	on success,
 *
 *\li		'rdatalist' is pointed to the rdatalist in rdataset.
 *
 * Returns:
 *\li	#ISC_R_SUCCESS
 */

ISC_LANG_ENDDECLS

#endif /* DNS_RDATALIST_H */
