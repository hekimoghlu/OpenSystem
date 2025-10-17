/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 12, 2024.
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
/* $Id: lwderror.c,v 1.12 2007/06/19 23:46:59 tbox Exp $ */

/*! \file */

#include <config.h>

#include <isc/socket.h>
#include <isc/util.h>

#include <named/types.h>
#include <named/lwdclient.h>

/*%
 * Generate an error packet for the client, schedule a send, and put us in
 * the SEND state.
 *
 * The client->pkt structure will be modified to form an error return.
 * The receiver needs to verify that it is in fact an error, and do the
 * right thing with it.  The opcode will be unchanged.  The result needs
 * to be set before calling this function.
 *
 * The only change this code makes is to set the receive buffer size to the
 * size we use, set the reply bit, and recompute any security information.
 */
void
ns_lwdclient_errorpktsend(ns_lwdclient_t *client, isc_uint32_t _result) {
	isc_result_t result;
	int lwres;
	isc_region_t r;
	lwres_buffer_t b;

	REQUIRE(NS_LWDCLIENT_ISRUNNING(client));

	/*
	 * Since we are only sending the packet header, we can safely toss
	 * the receive buffer.  This means we won't need to allocate space
	 * for sending an error reply.  This is a Good Thing.
	 */
	client->pkt.length = LWRES_LWPACKET_LENGTH;
	client->pkt.pktflags |= LWRES_LWPACKETFLAG_RESPONSE;
	client->pkt.recvlength = LWRES_RECVLENGTH;
	client->pkt.authtype = 0; /* XXXMLG */
	client->pkt.authlength = 0;
	client->pkt.result = _result;

	lwres_buffer_init(&b, client->buffer, LWRES_RECVLENGTH);
	lwres = lwres_lwpacket_renderheader(&b, &client->pkt);
	if (lwres != LWRES_R_SUCCESS) {
		ns_lwdclient_stateidle(client);
		return;
	}

	r.base = client->buffer;
	r.length = b.used;
	client->sendbuf = client->buffer;
	result = ns_lwdclient_sendreply(client, &r);
	if (result != ISC_R_SUCCESS) {
		ns_lwdclient_stateidle(client);
		return;
	}

	NS_LWDCLIENT_SETSEND(client);
}
