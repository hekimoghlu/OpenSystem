/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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
/* $Id: zonekey.c,v 1.9 2007/06/19 23:47:16 tbox Exp $ */

/*! \file */

#include <config.h>

#include <isc/result.h>
#include <isc/types.h>
#include <isc/util.h>

#include <dns/keyvalues.h>
#include <dns/rdata.h>
#include <dns/rdatastruct.h>
#include <dns/types.h>
#include <dns/zonekey.h>

isc_boolean_t
dns_zonekey_iszonekey(dns_rdata_t *keyrdata) {
	isc_result_t result;
	dns_rdata_dnskey_t key;
	isc_boolean_t iszonekey = ISC_TRUE;

	REQUIRE(keyrdata != NULL);

	result = dns_rdata_tostruct(keyrdata, &key, NULL);
	if (result != ISC_R_SUCCESS)
		return (ISC_FALSE);

	if ((key.flags & DNS_KEYTYPE_NOAUTH) != 0)
		iszonekey = ISC_FALSE;
	if ((key.flags & DNS_KEYFLAG_OWNERMASK) != DNS_KEYOWNER_ZONE)
		iszonekey = ISC_FALSE;
	if (key.protocol != DNS_KEYPROTO_DNSSEC &&
	key.protocol != DNS_KEYPROTO_ANY)
		iszonekey = ISC_FALSE;
	
	return (iszonekey);
}
