/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#ifndef __DNSINFO_H__
#define __DNSINFO_H__

/*
 * These routines provide access to the systems DNS configuration
 */

#include <os/availability.h>
#include <sys/cdefs.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define	DNSINFO_VERSION		20170629

#define DEFAULT_SEARCH_ORDER    200000   /* search order for the "default" resolver domain name */

#define	DNS_PTR(type, name)				\
	union {						\
		type		name;			\
		uint64_t	_ ## name ## _p;	\
	}

#define	DNS_VAR(type, name)				\
	type	name


#pragma pack(4)
typedef struct {
	struct in_addr	address;
	struct in_addr	mask;
} dns_sortaddr_t;
#pragma pack()


#pragma pack(4)
typedef struct {
	DNS_PTR(char *,			domain);	/* domain */
	DNS_VAR(int32_t,		n_nameserver);	/* # nameserver */
	DNS_PTR(struct sockaddr **,	nameserver);
	DNS_VAR(uint16_t,		port);		/* port (in host byte order) */
	DNS_VAR(int32_t,		n_search);	/* # search */
	DNS_PTR(char **,		search);
	DNS_VAR(int32_t,		n_sortaddr);	/* # sortaddr */
	DNS_PTR(dns_sortaddr_t **,	sortaddr);
	DNS_PTR(char *,			options);	/* options */
	DNS_VAR(uint32_t,		timeout);	/* timeout */
	DNS_VAR(uint32_t,		search_order);	/* search_order */
	DNS_VAR(uint32_t,		if_index);
	DNS_VAR(uint32_t,		flags);
	DNS_VAR(uint32_t,		reach_flags);	/* SCNetworkReachabilityFlags */
	DNS_VAR(uint32_t,		service_identifier);
	DNS_PTR(char *,			cid);		/* configuration identifer */
	DNS_PTR(char *,			if_name);	/* if_index interface name */
} dns_resolver_t;
#pragma pack()


#define DNS_RESOLVER_FLAGS_REQUEST_A_RECORDS	0x0002		/* always requesting for A dns records in queries */
#define DNS_RESOLVER_FLAGS_REQUEST_AAAA_RECORDS	0x0004		/* always requesting for AAAA dns records in queries */

#define	DNS_RESOLVER_FLAGS_REQUEST_ALL_RECORDS	\
	(DNS_RESOLVER_FLAGS_REQUEST_A_RECORDS | DNS_RESOLVER_FLAGS_REQUEST_AAAA_RECORDS)

#define DNS_RESOLVER_FLAGS_SCOPED		0x1000		/* configuration is for scoped questions */
#define DNS_RESOLVER_FLAGS_SERVICE_SPECIFIC	0x2000		/* configuration is service-specific */
#define DNS_RESOLVER_FLAGS_SUPPLEMENTAL		0x4000		/* supplemental match configuration */


#pragma pack(4)
typedef struct {
	DNS_VAR(int32_t,		n_resolver);		/* resolver configurations */
	DNS_PTR(dns_resolver_t **,	resolver);
	DNS_VAR(int32_t,		n_scoped_resolver);	/* "scoped" resolver configurations */
	DNS_PTR(dns_resolver_t **,	scoped_resolver);
	DNS_VAR(uint64_t,		generation);
	DNS_VAR(int32_t,		n_service_specific_resolver);
	DNS_PTR(dns_resolver_t **,	service_specific_resolver);
	DNS_VAR(uint32_t,		version);
} dns_config_t;
#pragma pack()


__BEGIN_DECLS

/*
 * DNS configuration access APIs
 */
const char *
dns_configuration_notify_key    (void)				API_AVAILABLE(macos(10.4), ios(2.0));

dns_config_t *
dns_configuration_copy		(void)				API_AVAILABLE(macos(10.4), ios(2.0));

void
dns_configuration_free		(dns_config_t	*config)	API_AVAILABLE(macos(10.4), ios(2.0));

void
_dns_configuration_ack		(dns_config_t	*config,
				 const char	*bundle_id)	API_AVAILABLE(macos(10.8), ios(6.0));

__END_DECLS

#endif	/* __DNSINFO_H__ */
