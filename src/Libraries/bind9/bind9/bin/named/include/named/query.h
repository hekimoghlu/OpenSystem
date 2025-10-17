/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
#ifndef NAMED_QUERY_H
#define NAMED_QUERY_H 1

/*! \file */

#include <isc/types.h>
#include <isc/buffer.h>
#include <isc/netaddr.h>

#include <dns/rdataset.h>
#include <dns/rpz.h>
#include <dns/types.h>

#include <named/types.h>

/*% nameserver database version structure */
typedef struct ns_dbversion {
	dns_db_t			*db;
	dns_dbversion_t			*version;
	isc_boolean_t			acl_checked;
	isc_boolean_t			queryok;
	ISC_LINK(struct ns_dbversion)	link;
} ns_dbversion_t;

/*% nameserver query structure */
struct ns_query {
	unsigned int			attributes;
	unsigned int			restarts;
	isc_boolean_t			timerset;
	dns_name_t *			qname;
	dns_name_t *			origqname;
	dns_rdatatype_t			qtype;
	unsigned int			dboptions;
	unsigned int			fetchoptions;
	dns_db_t *			gluedb;
	dns_db_t *			authdb;
	dns_zone_t *			authzone;
	isc_boolean_t			authdbset;
	isc_boolean_t			isreferral;
	isc_mutex_t			fetchlock;
	dns_fetch_t *			fetch;
	dns_fetch_t *			prefetch;
	dns_rpz_st_t *			rpz_st;
	isc_bufferlist_t		namebufs;
	ISC_LIST(ns_dbversion_t)	activeversions;
	ISC_LIST(ns_dbversion_t)	freeversions;
	dns_rdataset_t *		dns64_aaaa;
	dns_rdataset_t *		dns64_sigaaaa;
	isc_boolean_t *			dns64_aaaaok;
	unsigned int			dns64_aaaaoklen;
	unsigned int			dns64_options;
	unsigned int			dns64_ttl;
};

#define NS_QUERYATTR_RECURSIONOK	0x0001
#define NS_QUERYATTR_CACHEOK		0x0002
#define NS_QUERYATTR_PARTIALANSWER	0x0004
#define NS_QUERYATTR_NAMEBUFUSED	0x0008
#define NS_QUERYATTR_RECURSING		0x0010
#define NS_QUERYATTR_CACHEGLUEOK	0x0020
#define NS_QUERYATTR_QUERYOKVALID	0x0040
#define NS_QUERYATTR_QUERYOK		0x0080
#define NS_QUERYATTR_WANTRECURSION	0x0100
#define NS_QUERYATTR_SECURE		0x0200
#define NS_QUERYATTR_NOAUTHORITY	0x0400
#define NS_QUERYATTR_NOADDITIONAL	0x0800
#define NS_QUERYATTR_CACHEACLOKVALID	0x1000
#define NS_QUERYATTR_CACHEACLOK		0x2000
#define NS_QUERYATTR_DNS64		0x4000
#define NS_QUERYATTR_DNS64EXCLUDE	0x8000
#define NS_QUERYATTR_RRL_CHECKED	0x10000


isc_result_t
ns_query_init(ns_client_t *client);

void
ns_query_free(ns_client_t *client);

void
ns_query_start(ns_client_t *client);

void
ns_query_cancel(ns_client_t *client);

#endif /* NAMED_QUERY_H */
