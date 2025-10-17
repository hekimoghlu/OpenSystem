/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 8, 2023.
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
/* <rdar://problem/24849324> os_channel_{rx,tx}_ring() needs to check bounds of the ring index */

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <uuid/uuid.h>
#include <sys/select.h>
#include <poll.h>
#include <sys/event.h>
#include "skywalk_test_driver.h"
#include "skywalk_test_common.h"
#include "skywalk_test_utils.h"

#define NUM_TX_RINGS 4
#define NUM_RX_RINGS 4

static void
skt_ringid_init(void)
{
	struct sktc_nexus_attr attr = SKTC_NEXUS_ATTR_INIT();

	strncpy((char *)attr.name, "skywalk_test_ringid_upipe",
	    sizeof(nexus_name_t) - 1);
	attr.type = NEXUS_TYPE_USER_PIPE;
	attr.ntxrings = NUM_TX_RINGS;
	attr.nrxrings = NUM_RX_RINGS;
	attr.anonymous = 1;

	sktc_setup_nexus(&attr);
}

static void
skt_ringid_fini(void)
{
	sktc_cleanup_nexus();
}

/****************************************************************/

static int
skt_ringid_main_common(int argc, char *argv[], uint32_t num,
    ring_id_type_t first, ring_id_type_t last,
    channel_ring_t (*get_ring)(const channel_t chd, const ring_id_t rid))
{
	int error;
	channel_t channel;
	uuid_t channel_uuid;
	ring_id_t fringid, lringid, ringid;
	channel_ring_t ring;

	error = uuid_parse(argv[3], channel_uuid);
	SKTC_ASSERT_ERR(!error);

	channel = sktu_channel_create_extended(channel_uuid, 0,
	    CHANNEL_DIR_TX_RX, CHANNEL_RING_ID_ANY, NULL,
	    -1, -1, -1, -1, -1, -1, -1, 1, -1, -1);
	assert(channel);

	fringid = os_channel_ring_id(channel, first);
	lringid = os_channel_ring_id(channel, last);

	assert(lringid - fringid == num - 1);

	assert(fringid == 0); // XXX violates opaque abstraction

	/* Verify that we can get all the expected rings */
	for (ringid = fringid; ringid <= lringid; ringid++) {
		ring = (*get_ring)(channel, ringid);
		assert(ring);
	}

	/* And not a ring outside of the range */
	assert(ringid == lringid + 1);
	ring = (*get_ring)(channel, ringid);
	assert(!ring);

	os_channel_destroy(channel);

	/* Now reopen each channel with just a single ringid
	 * And verify that we can only get the expected ring id
	 */
	for (ringid = fringid; ringid <= lringid; ringid++) {
		ring_id_t ringid2;

		channel = sktu_channel_create_extended(channel_uuid, 0,
		    CHANNEL_DIR_TX_RX, ringid, NULL,
		    -1, -1, -1, -1, -1, -1, -1, 1, -1, -1);
		assert(channel);

		ringid2 = os_channel_ring_id(channel, first);
		assert(ringid2 == ringid);

		ringid2 = os_channel_ring_id(channel, last);
		assert(ringid2 == ringid);

		for (ringid2 = fringid; ringid2 <= lringid + 1; ringid2++) {
			ring = (*get_ring)(channel, ringid2);
			assert(ringid2 != ringid || ring);
			assert(ringid2 == ringid || !ring); // This verifies rdar://problem/24849324
		}

		os_channel_destroy(channel);
	}

	/* Now try to reopen the channel with an invalid ringid */
	assert(ringid == lringid + 1);
	channel = sktu_channel_create_extended(channel_uuid, 0,
	    CHANNEL_DIR_TX_RX, ringid, NULL,
	    -1, -1, -1, -1, -1, -1, -1, 1, -1, -1);
	assert(!channel);

	return 0;
}

/****************************************************************/

static int
skt_ringidtx_main(int argc, char *argv[])
{
	return skt_ringid_main_common(argc, argv,
	           NUM_TX_RINGS, CHANNEL_FIRST_TX_RING, CHANNEL_LAST_TX_RING,
	           &os_channel_tx_ring);
}

static int
skt_ringidrx_main(int argc, char *argv[])
{
	return skt_ringid_main_common(argc, argv,
	           NUM_RX_RINGS, CHANNEL_FIRST_RX_RING, CHANNEL_LAST_RX_RING,
	           &os_channel_rx_ring);
}

struct skywalk_test skt_ringidtx = {
	"ringidtx", "tests opening tx ringids",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_USER_PIPE,
	skt_ringidtx_main, SKTC_GENERIC_UPIPE_ARGV,
	skt_ringid_init, skt_ringid_fini,
};

struct skywalk_test skt_ringidrx = {
	"ringidrx", "tests opening rx ringids",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_USER_PIPE,
	skt_ringidrx_main, SKTC_GENERIC_UPIPE_ARGV,
	skt_ringid_init, skt_ringid_fini,
};

/****************************************************************/
