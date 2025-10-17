/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
/* $Id: portset.c,v 1.4 2008/06/24 23:24:35 marka Exp $ */

/*! \file */

#include <config.h>

#include <isc/mem.h>
#include <isc/portset.h>
#include <isc/string.h>
#include <isc/types.h>
#include <isc/util.h>

#define ISC_PORTSET_BUFSIZE (65536 / (sizeof(isc_uint32_t) * 8))

/*%
 * Internal representation of portset.  It's an array of 32-bit integers, each
 * bit corresponding to a single port in the ascending order.  For example,
 * the second most significant bit of buf[0] corresponds to port 1.
 */
struct isc_portset {
	unsigned int nports;	/*%< number of ports in the set */
	isc_uint32_t buf[ISC_PORTSET_BUFSIZE];
};

static inline isc_boolean_t
portset_isset(isc_portset_t *portset, in_port_t port) {
	return (ISC_TF((portset->buf[port >> 5] & (1 << (port & 31))) != 0));
}

static inline void
portset_add(isc_portset_t *portset, in_port_t port) {
	if (!portset_isset(portset, port)) {
		portset->nports++;
		portset->buf[port >> 5] |= (1 << (port & 31));
	}
}

static inline void
portset_remove(isc_portset_t *portset, in_port_t port) {
	if (portset_isset(portset, port)) {
		portset->nports--;
		portset->buf[port >> 5] &= ~(1 << (port & 31));
	}
}

isc_result_t
isc_portset_create(isc_mem_t *mctx, isc_portset_t **portsetp) {
	isc_portset_t *portset;

	REQUIRE(portsetp != NULL && *portsetp == NULL);

	portset = isc_mem_get(mctx, sizeof(*portset));
	if (portset == NULL)
		return (ISC_R_NOMEMORY);

	/* Make the set 'empty' by default */
	memset(portset, 0, sizeof(*portset));
	*portsetp = portset;

	return (ISC_R_SUCCESS);
}

void
isc_portset_destroy(isc_mem_t *mctx, isc_portset_t **portsetp) {
	isc_portset_t *portset;

	REQUIRE(portsetp != NULL);
	portset = *portsetp;

	isc_mem_put(mctx, portset, sizeof(*portset));
}

isc_boolean_t
isc_portset_isset(isc_portset_t *portset, in_port_t port) {
	REQUIRE(portset != NULL);

	return (portset_isset(portset, port));
}

unsigned int
isc_portset_nports(isc_portset_t *portset) {
	REQUIRE(portset != NULL);

	return (portset->nports);
}

void
isc_portset_add(isc_portset_t *portset, in_port_t port) {
	REQUIRE(portset != NULL);

	portset_add(portset, port);
}

void
isc_portset_remove(isc_portset_t *portset, in_port_t port) {
	portset_remove(portset, port);
}

void
isc_portset_addrange(isc_portset_t *portset, in_port_t port_lo,
		     in_port_t port_hi)
{
	in_port_t p;

	REQUIRE(portset != NULL);
	REQUIRE(port_lo <= port_hi);

	p = port_lo;
	do {
		portset_add(portset, p);
	} while (p++ < port_hi);
}

void
isc_portset_removerange(isc_portset_t *portset, in_port_t port_lo,
			in_port_t port_hi)
{
	in_port_t p;

	REQUIRE(portset != NULL);
	REQUIRE(port_lo <= port_hi);

	p = port_lo;
	do {
		portset_remove(portset, p);
	} while (p++ < port_hi);
}
