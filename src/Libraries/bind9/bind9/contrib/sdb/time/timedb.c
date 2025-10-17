/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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
/* $Id: timedb.c,v 1.12 2011/10/11 23:46:45 tbox Exp $ */

/*
 * A simple database driver that enables the server to return the
 * current time in a DNS record.
 */

#include <config.h>

#include <string.h>
#include <stdio.h>
#include <time.h>

#include <isc/print.h>
#include <isc/result.h>
#include <isc/util.h>

#include <dns/sdb.h>

#include <named/globals.h>

#include "timedb.h"

static dns_sdbimplementation_t *timedb = NULL;

/*
 * This database operates on relative names.
 *
 * "time" and "@" return the time in a TXT record.
 * "clock" is a CNAME to "time"
 * "current" is a DNAME to "@" (try time.current.time)
 */
#ifdef DNS_CLIENTINFO_VERSION
static isc_result_t
timedb_lookup(const char *zone, const char *name, void *dbdata,
	      dns_sdblookup_t *lookup, dns_clientinfomethods_t *methods,
	      dns_clientinfo_t *clientinfo)
#else
static isc_result_t
timedb_lookup(const char *zone, const char *name, void *dbdata,
	      dns_sdblookup_t *lookup)
#endif /* DNS_CLIENTINFO_VERSION */
{
	isc_result_t result;

	UNUSED(zone);
	UNUSED(dbdata);
#ifdef DNS_CLIENTINFO_VERSION
	UNUSED(methods);
	UNUSED(clientinfo);
#endif /* DNS_CLIENTINFO_VERSION */

	if (strcmp(name, "@") == 0 || strcmp(name, "time") == 0) {
		time_t now = time(NULL);
		char buf[100];
		int n;

		/*
		 * Call ctime to create the string, put it in quotes, and
		 * remove the trailing newline.
		 */
		n = snprintf(buf, sizeof(buf), "\"%s", ctime(&now));
		if (n < 0)
			return (ISC_R_FAILURE);
		buf[n - 1] = '\"';
		result = dns_sdb_putrr(lookup, "txt", 1, buf);
		if (result != ISC_R_SUCCESS)
			return (ISC_R_FAILURE);
	} else if (strcmp(name, "clock") == 0) {
		result = dns_sdb_putrr(lookup, "cname", 1, "time");
		if (result != ISC_R_SUCCESS)
			return (ISC_R_FAILURE);
	} else if (strcmp(name, "current") == 0) {
		result = dns_sdb_putrr(lookup, "dname", 1, "@");
		if (result != ISC_R_SUCCESS)
			return (ISC_R_FAILURE);
	} else
		return (ISC_R_NOTFOUND);

	return (ISC_R_SUCCESS);
}

/*
 * lookup() does not return SOA or NS records, so authority() must be defined.
 */
static isc_result_t
timedb_authority(const char *zone, void *dbdata, dns_sdblookup_t *lookup) {
	isc_result_t result;

	UNUSED(zone);
	UNUSED(dbdata);

	result = dns_sdb_putsoa(lookup, "localhost.", "root.localhost.", 0);
	if (result != ISC_R_SUCCESS)
		return (ISC_R_FAILURE);

	result = dns_sdb_putrr(lookup, "ns", 86400, "ns1.localdomain.");
	if (result != ISC_R_SUCCESS)
		return (ISC_R_FAILURE);
	result = dns_sdb_putrr(lookup, "ns", 86400, "ns2.localdomain.");
	if (result != ISC_R_SUCCESS)
		return (ISC_R_FAILURE);

	return (ISC_R_SUCCESS);
}

/*
 * This zone does not support zone transfer, so allnodes() is NULL.  There
 * is no database specific data, so create() and destroy() are NULL.
 */
static dns_sdbmethods_t timedb_methods = {
	timedb_lookup,
	timedb_authority,
	NULL,	/* allnodes */
	NULL,	/* create */
	NULL,	/* destroy */
	NULL	/* lookup2 */
};

/*
 * Wrapper around dns_sdb_register().
 */
isc_result_t
timedb_init(void) {
	unsigned int flags;
	flags = DNS_SDBFLAG_RELATIVEOWNER | DNS_SDBFLAG_RELATIVERDATA;
	return (dns_sdb_register("time", &timedb_methods, NULL, flags,
				 ns_g_mctx, &timedb));
}

/*
 * Wrapper around dns_sdb_unregister().
 */
void
timedb_clear(void) {
	if (timedb != NULL)
		dns_sdb_unregister(&timedb);
}
