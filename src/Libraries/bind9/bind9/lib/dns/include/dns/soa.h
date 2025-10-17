/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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
/* $Id: soa.h,v 1.12 2009/09/10 01:47:09 each Exp $ */

#ifndef DNS_SOA_H
#define DNS_SOA_H 1

/*****
 ***** Module Info
 *****/

/*! \file dns/soa.h
 * \brief
 * SOA utilities.
 */

/***
 *** Imports
 ***/

#include <isc/lang.h>
#include <isc/types.h>

#include <dns/types.h>

ISC_LANG_BEGINDECLS

#define DNS_SOA_BUFFERSIZE      ((2 * DNS_NAME_MAXWIRE) + (4 * 5))

isc_result_t
dns_soa_buildrdata(dns_name_t *origin, dns_name_t *contact,
		   dns_rdataclass_t rdclass,
		   isc_uint32_t serial, isc_uint32_t refresh,
		   isc_uint32_t retry, isc_uint32_t expire,
		   isc_uint32_t minimum, unsigned char *buffer,
		   dns_rdata_t *rdata);
/*%<
 * Build the rdata of an SOA record.
 *
 * Requires:
 *\li   buffer  Points to a temporary buffer of at least
 *              DNS_SOA_BUFFERSIZE bytes.
 *\li   rdata   Points to an initialized dns_rdata_t.
 *
 * Ensures:
 *  \li    *rdata       Contains a valid SOA rdata.  The 'data' member
 *  			refers to 'buffer'.
 */

isc_uint32_t
dns_soa_getserial(dns_rdata_t *rdata);
isc_uint32_t
dns_soa_getrefresh(dns_rdata_t *rdata);
isc_uint32_t
dns_soa_getretry(dns_rdata_t *rdata);
isc_uint32_t
dns_soa_getexpire(dns_rdata_t *rdata);
isc_uint32_t
dns_soa_getminimum(dns_rdata_t *rdata);
/*
 * Extract an integer field from the rdata of a SOA record.
 *
 * Requires:
 *	rdata refers to the rdata of a well-formed SOA record.
 */

void
dns_soa_setserial(isc_uint32_t val, dns_rdata_t *rdata);
void
dns_soa_setrefresh(isc_uint32_t val, dns_rdata_t *rdata);
void
dns_soa_setretry(isc_uint32_t val, dns_rdata_t *rdata);
void
dns_soa_setexpire(isc_uint32_t val, dns_rdata_t *rdata);
void
dns_soa_setminimum(isc_uint32_t val, dns_rdata_t *rdata);
/*
 * Change an integer field of a SOA record by modifying the
 * rdata in-place.
 *
 * Requires:
 *	rdata refers to the rdata of a well-formed SOA record.
 */


ISC_LANG_ENDDECLS

#endif /* DNS_SOA_H */
