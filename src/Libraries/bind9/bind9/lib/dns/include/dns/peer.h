/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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
/* $Id: peer.h,v 1.35 2009/01/17 23:47:43 tbox Exp $ */

#ifndef DNS_PEER_H
#define DNS_PEER_H 1

/*****
 ***** Module Info
 *****/

/*! \file dns/peer.h
 * \brief
 * Data structures for peers (e.g. a 'server' config file statement)
 */

/***
 *** Imports
 ***/

#include <isc/lang.h>
#include <isc/magic.h>
#include <isc/netaddr.h>

#include <dns/types.h>

#define DNS_PEERLIST_MAGIC	ISC_MAGIC('s','e','R','L')
#define DNS_PEER_MAGIC		ISC_MAGIC('S','E','r','v')

#define DNS_PEERLIST_VALID(ptr)	ISC_MAGIC_VALID(ptr, DNS_PEERLIST_MAGIC)
#define DNS_PEER_VALID(ptr)	ISC_MAGIC_VALID(ptr, DNS_PEER_MAGIC)

/***
 *** Types
 ***/

struct dns_peerlist {
	unsigned int		magic;
	isc_uint32_t		refs;

	isc_mem_t	       *mem;

	ISC_LIST(dns_peer_t) elements;
};

struct dns_peer {
	unsigned int		magic;
	isc_uint32_t		refs;

	isc_mem_t	       *mem;

	isc_netaddr_t		address;
	unsigned int		prefixlen;
	isc_boolean_t		bogus;
	dns_transfer_format_t	transfer_format;
	isc_uint32_t		transfers;
	isc_boolean_t		support_ixfr;
	isc_boolean_t		provide_ixfr;
	isc_boolean_t		request_ixfr;
	isc_boolean_t		support_edns;
	isc_boolean_t		request_nsid;
	isc_boolean_t		request_sit;
	isc_boolean_t		force_tcp;
	dns_name_t	       *key;
	isc_sockaddr_t	       *transfer_source;
	isc_dscp_t		transfer_dscp;
	isc_sockaddr_t	       *notify_source;
	isc_dscp_t		notify_dscp;
	isc_sockaddr_t	       *query_source;
	isc_dscp_t		query_dscp;
	isc_uint16_t		udpsize;		/* receive size */
	isc_uint16_t		maxudp;			/* transmit size */

	isc_uint32_t		bitflags;

	ISC_LINK(dns_peer_t)	next;
};

/***
 *** Functions
 ***/

ISC_LANG_BEGINDECLS

isc_result_t
dns_peerlist_new(isc_mem_t *mem, dns_peerlist_t **list);

void
dns_peerlist_attach(dns_peerlist_t *source, dns_peerlist_t **target);

void
dns_peerlist_detach(dns_peerlist_t **list);

/*
 * After return caller still holds a reference to peer.
 */
void
dns_peerlist_addpeer(dns_peerlist_t *peers, dns_peer_t *peer);

/*
 * Ditto. */
isc_result_t
dns_peerlist_peerbyaddr(dns_peerlist_t *peers, isc_netaddr_t *addr,
			dns_peer_t **retval);

/*
 * What he said.
 */
isc_result_t
dns_peerlist_currpeer(dns_peerlist_t *peers, dns_peer_t **retval);

isc_result_t
dns_peer_new(isc_mem_t *mem, isc_netaddr_t *ipaddr, dns_peer_t **peer);

isc_result_t
dns_peer_newprefix(isc_mem_t *mem, isc_netaddr_t *ipaddr,
		   unsigned int prefixlen, dns_peer_t **peer);

void
dns_peer_attach(dns_peer_t *source, dns_peer_t **target);

void
dns_peer_detach(dns_peer_t **list);

isc_result_t
dns_peer_setbogus(dns_peer_t *peer, isc_boolean_t newval);

isc_result_t
dns_peer_getbogus(dns_peer_t *peer, isc_boolean_t *retval);

isc_result_t
dns_peer_setrequestixfr(dns_peer_t *peer, isc_boolean_t newval);

isc_result_t
dns_peer_getrequestixfr(dns_peer_t *peer, isc_boolean_t *retval);

isc_result_t
dns_peer_setprovideixfr(dns_peer_t *peer, isc_boolean_t newval);

isc_result_t
dns_peer_getprovideixfr(dns_peer_t *peer, isc_boolean_t *retval);

isc_result_t
dns_peer_setrequestnsid(dns_peer_t *peer, isc_boolean_t newval);

isc_result_t
dns_peer_getrequestnsid(dns_peer_t *peer, isc_boolean_t *retval);

isc_result_t
dns_peer_setrequestsit(dns_peer_t *peer, isc_boolean_t newval);

isc_result_t
dns_peer_getrequestsit(dns_peer_t *peer, isc_boolean_t *retval);

isc_result_t
dns_peer_setforcetcp(dns_peer_t *peer, isc_boolean_t newval);

isc_result_t
dns_peer_getforcetcp(dns_peer_t *peer, isc_boolean_t *retval);

isc_result_t
dns_peer_setsupportedns(dns_peer_t *peer, isc_boolean_t newval);

isc_result_t
dns_peer_getsupportedns(dns_peer_t *peer, isc_boolean_t *retval);

isc_result_t
dns_peer_settransfers(dns_peer_t *peer, isc_uint32_t newval);

isc_result_t
dns_peer_gettransfers(dns_peer_t *peer, isc_uint32_t *retval);

isc_result_t
dns_peer_settransferformat(dns_peer_t *peer, dns_transfer_format_t newval);

isc_result_t
dns_peer_gettransferformat(dns_peer_t *peer, dns_transfer_format_t *retval);

isc_result_t
dns_peer_setkeybycharp(dns_peer_t *peer, const char *keyval);

isc_result_t
dns_peer_getkey(dns_peer_t *peer, dns_name_t **retval);

isc_result_t
dns_peer_setkey(dns_peer_t *peer, dns_name_t **keyval);

isc_result_t
dns_peer_settransfersource(dns_peer_t *peer,
			   const isc_sockaddr_t *transfer_source);

isc_result_t
dns_peer_gettransfersource(dns_peer_t *peer, isc_sockaddr_t *transfer_source);

isc_result_t
dns_peer_setudpsize(dns_peer_t *peer, isc_uint16_t udpsize);

isc_result_t
dns_peer_getudpsize(dns_peer_t *peer, isc_uint16_t *udpsize);

isc_result_t
dns_peer_setmaxudp(dns_peer_t *peer, isc_uint16_t maxudp);

isc_result_t
dns_peer_getmaxudp(dns_peer_t *peer, isc_uint16_t *maxudp);

isc_result_t
dns_peer_setnotifysource(dns_peer_t *peer, const isc_sockaddr_t *notify_source);

isc_result_t
dns_peer_getnotifysource(dns_peer_t *peer, isc_sockaddr_t *notify_source);

isc_result_t
dns_peer_setquerysource(dns_peer_t *peer, const isc_sockaddr_t *query_source);

isc_result_t
dns_peer_getquerysource(dns_peer_t *peer, isc_sockaddr_t *query_source);

isc_result_t
dns_peer_setnotifydscp(dns_peer_t *peer, isc_dscp_t dscp);

isc_result_t
dns_peer_getnotifydscp(dns_peer_t *peer, isc_dscp_t *dscpp);

isc_result_t
dns_peer_settransferdscp(dns_peer_t *peer, isc_dscp_t dscp);

isc_result_t
dns_peer_gettransferdscp(dns_peer_t *peer, isc_dscp_t *dscpp);

isc_result_t
dns_peer_setquerydscp(dns_peer_t *peer, isc_dscp_t dscp);

isc_result_t
dns_peer_getquerydscp(dns_peer_t *peer, isc_dscp_t *dscpp);
ISC_LANG_ENDDECLS

#endif /* DNS_PEER_H */
