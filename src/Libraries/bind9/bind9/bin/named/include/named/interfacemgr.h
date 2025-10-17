/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
/* $Id: interfacemgr.h,v 1.35 2011/07/28 23:47:58 tbox Exp $ */

#ifndef NAMED_INTERFACEMGR_H
#define NAMED_INTERFACEMGR_H 1

/*****
 ***** Module Info
 *****/

/*! \file
 * \brief
 * The interface manager monitors the operating system's list
 * of network interfaces, creating and destroying listeners
 * as needed.
 *
 * Reliability:
 *\li	No impact expected.
 *
 * Resources:
 *
 * Security:
 * \li	The server will only be able to bind to the DNS port on
 *	newly discovered interfaces if it is running as root.
 *
 * Standards:
 *\li	The API for scanning varies greatly among operating systems.
 *	This module attempts to hide the differences.
 */

/***
 *** Imports
 ***/

#include <isc/magic.h>
#include <isc/mem.h>
#include <isc/socket.h>

#include <dns/result.h>

#include <named/listenlist.h>
#include <named/types.h>

/***
 *** Types
 ***/

#define IFACE_MAGIC		ISC_MAGIC('I',':','-',')')
#define NS_INTERFACE_VALID(t)	ISC_MAGIC_VALID(t, IFACE_MAGIC)

#define NS_INTERFACEFLAG_ANYADDR	0x01U	/*%< bound to "any" address */
#define MAX_UDP_DISPATCH 128		/*%< Maximum number of UDP dispatchers
						     to start per interface */
/*% The nameserver interface structure */
struct ns_interface {
	unsigned int		magic;		/*%< Magic number. */
	ns_interfacemgr_t *	mgr;		/*%< Interface manager. */
	isc_mutex_t		lock;
	int			references;	/*%< Locked */
	unsigned int		generation;     /*%< Generation number. */
	isc_sockaddr_t		addr;           /*%< Address and port. */
	unsigned int		flags;		/*%< Interface characteristics */
	char 			name[32];	/*%< Null terminated. */
	dns_dispatch_t *	udpdispatch[MAX_UDP_DISPATCH];
						/*%< UDP dispatchers. */
	isc_socket_t *		tcpsocket;	/*%< TCP socket. */
	isc_dscp_t		dscp;		/*%< "listen-on" DSCP value */
	int			ntcptarget;	/*%< Desired number of concurrent
						     TCP accepts */
	int			ntcpcurrent;	/*%< Current ditto, locked */
	int			nudpdispatch;	/*%< Number of UDP dispatches */
	ns_clientmgr_t *	clientmgr;	/*%< Client manager. */
	ISC_LINK(ns_interface_t) link;
};

/***
 *** Functions
 ***/

isc_result_t
ns_interfacemgr_create(isc_mem_t *mctx, isc_taskmgr_t *taskmgr,
		       isc_socketmgr_t *socketmgr,
		       dns_dispatchmgr_t *dispatchmgr,
		       isc_task_t *task, ns_interfacemgr_t **mgrp);
/*%
 * Create a new interface manager.
 *
 * Initially, the new manager will not listen on any interfaces.
 * Call ns_interfacemgr_setlistenon() and/or ns_interfacemgr_setlistenon6()
 * to set nonempty listen-on lists.
 */

void
ns_interfacemgr_attach(ns_interfacemgr_t *source, ns_interfacemgr_t **target);

void
ns_interfacemgr_detach(ns_interfacemgr_t **targetp);

void
ns_interfacemgr_shutdown(ns_interfacemgr_t *mgr);

void
ns_interfacemgr_scan(ns_interfacemgr_t *mgr, isc_boolean_t verbose);
/*%
 * Scan the operatings system's list of network interfaces
 * and create listeners when new interfaces are discovered.
 * Shut down the sockets for interfaces that go away.
 *
 * This should be called once on server startup and then
 * periodically according to the 'interface-interval' option
 * in named.conf.
 */

void
ns_interfacemgr_adjust(ns_interfacemgr_t *mgr, ns_listenlist_t *list,
		       isc_boolean_t verbose);
/*%
 * Similar to ns_interfacemgr_scan(), but this function also tries to see the
 * need for an explicit listen-on when a list element in 'list' is going to
 * override an already-listening a wildcard interface.
 *
 * This function does not update localhost and localnets ACLs.
 *
 * This should be called once on server startup, after configuring views and
 * zones.
 */

void
ns_interfacemgr_setlistenon4(ns_interfacemgr_t *mgr, ns_listenlist_t *value);
/*%
 * Set the IPv4 "listen-on" list of 'mgr' to 'value'.
 * The previous IPv4 listen-on list is freed.
 */

void
ns_interfacemgr_setlistenon6(ns_interfacemgr_t *mgr, ns_listenlist_t *value);
/*%
 * Set the IPv6 "listen-on" list of 'mgr' to 'value'.
 * The previous IPv6 listen-on list is freed.
 */

dns_aclenv_t *
ns_interfacemgr_getaclenv(ns_interfacemgr_t *mgr);

void
ns_interface_attach(ns_interface_t *source, ns_interface_t **target);

void
ns_interface_detach(ns_interface_t **targetp);

void
ns_interface_shutdown(ns_interface_t *ifp);
/*%
 * Stop listening for queries on interface 'ifp'.
 * May safely be called multiple times.
 */

void
ns_interfacemgr_dumprecursing(FILE *f, ns_interfacemgr_t *mgr);

isc_boolean_t
ns_interfacemgr_listeningon(ns_interfacemgr_t *mgr, isc_sockaddr_t *addr);

#endif /* NAMED_INTERFACEMGR_H */
