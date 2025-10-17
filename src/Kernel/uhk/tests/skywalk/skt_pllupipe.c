/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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
/* Attempts to run upipes in parallel.
 * Every upipe endpipe gets a thread
 * Todo: create variant that uses multiple rings per channel.
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <uuid/uuid.h>
#include <sys/select.h>
#include <poll.h>
#include <sys/event.h>
#include <darwintest.h>
#include "skywalk_test_driver.h"
#include "skywalk_test_common.h"
#include "skywalk_test_utils.h"

static int
skt_pllupipe_main(int argc, char *argv[])
{
	int error;
	uuid_t channel_uuid;
	int len = 10;

	error = uuid_parse(argv[3], channel_uuid);
	SKTC_ASSERT_ERR(!error);

	sktc_setup_channel_worker(channel_uuid, 0, CHANNEL_RING_ID_ANY,
	    NULL, 0, false, true);

	for (int i = 0; i < len; i++) {
		sleep(1);
		T_LOG("time %d of %d\n", i, len);
	}

	sktc_cleanup_channel_worker();

	return 0;
}

#define MMSLOTS 10000000

static int
skt_pllupipe_txk_main(int argc, char *argv[])
{
	int error;
	channel_t channel;
	ring_id_t ringid;
	channel_ring_t ring;
	uuid_t channel_uuid;

	error = uuid_parse(argv[3], channel_uuid);
	SKTC_ASSERT_ERR(!error);

	channel = sktu_channel_create_extended(channel_uuid, 0,
	    CHANNEL_DIR_TX_RX, CHANNEL_RING_ID_ANY, NULL,
	    -1, -1, -1, -1, -1, -1, -1, 1, -1, -1);
	assert(channel);

	ringid = os_channel_ring_id(channel, CHANNEL_FIRST_TX_RING);
	ring = os_channel_tx_ring(channel, ringid);
	assert(ring);

	sktc_pump_ring_nslots_kq(channel, ring, CHANNEL_SYNC_TX, true, MMSLOTS, true);

	return 0;
}

static int
skt_pllupipe_txs_main(int argc, char *argv[])
{
	int error;
	channel_t channel;
	ring_id_t ringid;
	channel_ring_t ring;
	uuid_t channel_uuid;

	error = uuid_parse(argv[3], channel_uuid);
	SKTC_ASSERT_ERR(!error);

	channel = sktu_channel_create_extended(channel_uuid, 0,
	    CHANNEL_DIR_TX_RX, CHANNEL_RING_ID_ANY, NULL,
	    -1, -1, -1, -1, -1, -1, -1, 1, -1, -1);
	assert(channel);

	ringid = os_channel_ring_id(channel, CHANNEL_FIRST_TX_RING);
	ring = os_channel_tx_ring(channel, ringid);
	assert(ring);

	sktc_pump_ring_nslots_select(channel, ring, CHANNEL_SYNC_TX, true, MMSLOTS, true);

	return 0;
}

static int
skt_pllupipe_txp_main(int argc, char *argv[])
{
	int error;
	channel_t channel;
	ring_id_t ringid;
	channel_ring_t ring;
	uuid_t channel_uuid;

	error = uuid_parse(argv[3], channel_uuid);
	SKTC_ASSERT_ERR(!error);

	channel = sktu_channel_create_extended(channel_uuid, 0,
	    CHANNEL_DIR_TX_RX, CHANNEL_RING_ID_ANY, NULL,
	    -1, -1, -1, -1, -1, -1, -1, 1, -1, -1);
	assert(channel);

	ringid = os_channel_ring_id(channel, CHANNEL_FIRST_TX_RING);
	ring = os_channel_tx_ring(channel, ringid);
	assert(ring);

	sktc_pump_ring_nslots_poll(channel, ring, CHANNEL_SYNC_TX, true, MMSLOTS, true);

	return 0;
}

static int
skt_pllupipe_rxk_main(int argc, char *argv[])
{
	int error;
	channel_t channel;
	ring_id_t ringid;
	channel_ring_t ring;
	uuid_t channel_uuid;

	error = uuid_parse(argv[3], channel_uuid);
	SKTC_ASSERT_ERR(!error);

	channel = sktu_channel_create_extended(channel_uuid, 0,
	    CHANNEL_DIR_TX_RX, CHANNEL_RING_ID_ANY, NULL,
	    -1, -1, -1, -1, -1, -1, -1, 1, -1, -1);
	assert(channel);

	ringid = os_channel_ring_id(channel, CHANNEL_FIRST_RX_RING);
	ring = os_channel_rx_ring(channel, ringid);
	assert(ring);

	sktc_pump_ring_nslots_kq(channel, ring, CHANNEL_SYNC_RX, true, MMSLOTS, true);

	return 0;
}

static int
skt_pllupipe_rxs_main(int argc, char *argv[])
{
	int error;
	channel_t channel;
	ring_id_t ringid;
	channel_ring_t ring;
	uuid_t channel_uuid;

	error = uuid_parse(argv[3], channel_uuid);
	SKTC_ASSERT_ERR(!error);

	channel = sktu_channel_create_extended(channel_uuid, 0,
	    CHANNEL_DIR_TX_RX, CHANNEL_RING_ID_ANY, NULL,
	    -1, -1, -1, -1, -1, -1, -1, 1, -1, -1);
	assert(channel);

	ringid = os_channel_ring_id(channel, CHANNEL_FIRST_RX_RING);
	ring = os_channel_rx_ring(channel, ringid);
	assert(ring);

	sktc_pump_ring_nslots_select(channel, ring, CHANNEL_SYNC_RX, true, MMSLOTS, true);

	return 0;
}

static int
skt_pllupipe_rxp_main(int argc, char *argv[])
{
	int error;
	channel_t channel;
	ring_id_t ringid;
	channel_ring_t ring;
	uuid_t channel_uuid;

	error = uuid_parse(argv[3], channel_uuid);
	SKTC_ASSERT_ERR(!error);

	channel = sktu_channel_create_extended(channel_uuid, 0,
	    CHANNEL_DIR_TX_RX, CHANNEL_RING_ID_ANY, NULL,
	    -1, -1, -1, -1, -1, -1, -1, 1, -1, -1);
	assert(channel);

	ringid = os_channel_ring_id(channel, CHANNEL_FIRST_RX_RING);
	ring = os_channel_rx_ring(channel, ringid);
	assert(ring);

	sktc_pump_ring_nslots_poll(channel, ring, CHANNEL_SYNC_RX, true, MMSLOTS, true);

	return 0;
}

struct skywalk_test skt_pllupipe = {
	"pllupipe", "fully parallel upipe for 10 seconds",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_USER_PIPE,
	skt_pllupipe_main, SKTC_GENERIC_UPIPE_ARGV,
	sktc_generic_upipe_null_init, sktc_generic_upipe_fini,
};

struct skywalk_test skt_pllutxk = {
	"pllutxk", "send 10000000 slots to upipe sink using kqueue",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_USER_PIPE,
	skt_pllupipe_txk_main, SKTC_GENERIC_UPIPE_ARGV,
	sktc_generic_upipe_null_init, sktc_generic_upipe_fini,
};

struct skywalk_test skt_pllutxs = {
	"pllutxs", "send 10000000 slots to upipe sink using select",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_USER_PIPE,
	skt_pllupipe_txs_main, SKTC_GENERIC_UPIPE_ARGV,
	sktc_generic_upipe_null_init, sktc_generic_upipe_fini,
};

struct skywalk_test skt_pllutxp = {
	"pllutxp", "send 10000000 slots to upipe sink using poll",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_USER_PIPE,
	skt_pllupipe_txp_main, SKTC_GENERIC_UPIPE_ARGV,
	sktc_generic_upipe_null_init, sktc_generic_upipe_fini,
};

struct skywalk_test skt_pllurxk = {
	"pllurxk", "receive 10000000 slots from upipe source using kqueue",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_USER_PIPE,
	skt_pllupipe_rxk_main, SKTC_GENERIC_UPIPE_ARGV,
	sktc_generic_upipe_null_init, sktc_generic_upipe_fini,
};

struct skywalk_test skt_pllurxs = {
	"pllurxs", "receive 10000000 slots from upipe source using select",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_USER_PIPE,
	skt_pllupipe_rxs_main, SKTC_GENERIC_UPIPE_ARGV,
	sktc_generic_upipe_null_init, sktc_generic_upipe_fini,
};

struct skywalk_test skt_pllurxp = {
	"pllurxp", "receive 10000000 slots to upipe source using poll",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_USER_PIPE,
	skt_pllupipe_rxp_main, SKTC_GENERIC_UPIPE_ARGV,
	sktc_generic_upipe_null_init, sktc_generic_upipe_fini,
};
