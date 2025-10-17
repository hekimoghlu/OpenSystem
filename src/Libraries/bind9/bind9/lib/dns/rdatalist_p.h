/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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
/* $Id: rdatalist_p.h,v 1.11 2008/09/25 04:02:38 tbox Exp $ */

#ifndef DNS_RDATALIST_P_H
#define DNS_RDATALIST_P_H

/*! \file */

#include <isc/result.h>
#include <dns/types.h>

ISC_LANG_BEGINDECLS

void
isc__rdatalist_disassociate(dns_rdataset_t *rdatasetp);

isc_result_t
isc__rdatalist_first(dns_rdataset_t *rdataset);

isc_result_t
isc__rdatalist_next(dns_rdataset_t *rdataset);

void
isc__rdatalist_current(dns_rdataset_t *rdataset, dns_rdata_t *rdata);

void
isc__rdatalist_clone(dns_rdataset_t *source, dns_rdataset_t *target);

unsigned int
isc__rdatalist_count(dns_rdataset_t *rdataset);

isc_result_t
isc__rdatalist_addnoqname(dns_rdataset_t *rdataset, dns_name_t *name);

isc_result_t
isc__rdatalist_getnoqname(dns_rdataset_t *rdataset, dns_name_t *name,
			  dns_rdataset_t *neg, dns_rdataset_t *negsig);

isc_result_t
isc__rdatalist_addclosest(dns_rdataset_t *rdataset, dns_name_t *name);

isc_result_t
isc__rdatalist_getclosest(dns_rdataset_t *rdataset, dns_name_t *name,
			  dns_rdataset_t *neg, dns_rdataset_t *negsig);

ISC_LANG_ENDDECLS

#endif /* DNS_RDATALIST_P_H */
