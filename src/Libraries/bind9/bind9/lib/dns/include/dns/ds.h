/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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
/* $Id: ds.h,v 1.12 2010/12/23 23:47:08 tbox Exp $ */

#ifndef DNS_DS_H
#define DNS_DS_H 1

#include <isc/lang.h>

#include <dns/types.h>

#define DNS_DSDIGEST_SHA1 (1)
#define DNS_DSDIGEST_SHA256 (2)
#define DNS_DSDIGEST_GOST (3)
#define DNS_DSDIGEST_SHA384 (4)

/*
 * Assuming SHA-384 digest type.
 */
#define DNS_DS_BUFFERSIZE (52)

ISC_LANG_BEGINDECLS

isc_result_t
dns_ds_buildrdata(dns_name_t *owner, dns_rdata_t *key,
		  unsigned int digest_type, unsigned char *buffer,
		  dns_rdata_t *rdata);
/*%<
 * Build the rdata of a DS record.
 *
 * Requires:
 *\li	key	Points to a valid DNS KEY record.
 *\li	buffer	Points to a temporary buffer of at least
 * 		#DNS_DS_BUFFERSIZE bytes.
 *\li	rdata	Points to an initialized dns_rdata_t.
 *
 * Ensures:
 *  \li    *rdata	Contains a valid DS rdata.  The 'data' member refers
 *		to 'buffer'.
 */

ISC_LANG_ENDDECLS

#endif /* DNS_DS_H */
