/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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
/* $Id: update.h,v 1.5 2011/08/30 23:46:53 tbox Exp $ */

#ifndef DNS_UPDATE_H
#define DNS_UPDATE_H 1

/*! \file dns/update.h */

/***
 ***	Imports
 ***/

#include <isc/lang.h>

#include <dns/types.h>
#include <dns/diff.h>

typedef struct {
	void (*func)(void *arg, dns_zone_t *zone, int level,
		     const char *message);
	void *arg;
} dns_update_log_t;

ISC_LANG_BEGINDECLS

/***
 ***	Functions
 ***/

isc_uint32_t
dns_update_soaserial(isc_uint32_t serial, dns_updatemethod_t method);
/*%<
 * Return the next serial number after 'serial', depending on the
 * update method 'method':
 *
 *\li	* dns_updatemethod_increment increments the serial number by one
 *\li	* dns_updatemethod_unixtime sets the serial number to the current
 *	  time (seconds since UNIX epoch) if possible, or increments by one
 *	  if not.
 */

isc_result_t
dns_update_signatures(dns_update_log_t *log, dns_zone_t *zone, dns_db_t *db,
		      dns_dbversion_t *oldver, dns_dbversion_t *newver,
		      dns_diff_t *diff, isc_uint32_t sigvalidityinterval);

isc_result_t
dns_update_signaturesinc(dns_update_log_t *log, dns_zone_t *zone, dns_db_t *db,
			 dns_dbversion_t *oldver, dns_dbversion_t *newver,
			 dns_diff_t *diff, isc_uint32_t sigvalidityinterval,
			 dns_update_state_t **state);

ISC_LANG_ENDDECLS

#endif /* DNS_UPDATE_H */
