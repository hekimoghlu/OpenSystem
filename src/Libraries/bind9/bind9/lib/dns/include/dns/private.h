/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 21, 2022.
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
/* $Id: private.h,v 1.5 2011/10/28 12:20:31 tbox Exp $ */

#include <isc/lang.h>
#include <isc/types.h>

#include <dns/types.h>
#include <dns/db.h>

#ifndef DNS_PRIVATE_H
#define DNS_PRIVATE_H

ISC_LANG_BEGINDECLS

isc_result_t
dns_private_chains(dns_db_t *db, dns_dbversion_t *ver,
		   dns_rdatatype_t privatetype,
		   isc_boolean_t *build_nsec, isc_boolean_t *build_nsec3);
/*%<
 * Examine the NSEC, NSEC3PARAM and privatetype RRsets at the apex of the
 * database to determine which of NSEC or NSEC3 chains we are currently
 * maintaining.  In normal operations only one of NSEC or NSEC3 is being
 * maintained but when we are transitiong between NSEC and NSEC3 we need
 * to update both sets of chains.  If 'privatetype' is zero then the
 * privatetype RRset will not be examined.
 *
 * Requires:
 * \li	'db' is valid.
 * \li	'version' is valid or NULL.
 * \li	'build_nsec' is a pointer to a isc_boolean_t or NULL.
 * \li	'build_nsec3' is a pointer to a isc_boolean_t or NULL.
 *
 * Returns:
 * \li 	ISC_R_SUCCESS, 'build_nsec' and 'build_nsec3' will be valid.
 * \li	other on error
 */

isc_result_t
dns_private_totext(dns_rdata_t *privaterdata, isc_buffer_t *buffer);
/*%<
 * Convert a private-type RR 'privaterdata' to human-readable form,
 * and place the result in 'buffer'.  The text should indicate
 * which action the private-type record specifies and whether the
 * action has been completed.
 *
 * Requires:
 * \li	'privaterdata' is a valid rdata containing at least five bytes
 * \li	'buffer' is a valid buffer
 *
 * Returns:
 * \li 	ISC_R_SUCCESS
 * \li	other on error
 */

ISC_LANG_ENDDECLS

#endif
