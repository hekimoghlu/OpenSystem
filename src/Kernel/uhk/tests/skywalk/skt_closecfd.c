/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <uuid/uuid.h>
#include <unistd.h>
#include "skywalk_test_driver.h"
#include "skywalk_test_common.h"
#include "skywalk_test_utils.h"

static int
skt_closecfd_main(int argc, char *argv[])
{
	int error;
	channel_t channel;
	uuid_t channel_uuid;
	int channelfd;

	error = uuid_parse(argv[3], channel_uuid);
	SKTC_ASSERT_ERR(!error);

	channel = sktu_channel_create_extended(channel_uuid, 0,
	    CHANNEL_DIR_TX_RX, CHANNEL_RING_ID_ANY, NULL,
	    -1, -1, -1, -1, -1, -1, -1, 1, -1, -1);
	assert(channel);

	channelfd = os_channel_get_fd(channel);
	assert(channelfd != -1);

	error = close(channelfd); // expected guarded fd fail.
	SKTC_ASSERT_ERR(!error);

	return 1; // should not reach
}

struct skywalk_test skt_closecfd = {
	"closecfd", "test closing guarded channel fd",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_USER_PIPE,
	skt_closecfd_main, SKTC_GENERIC_UPIPE_ARGV,
	sktc_generic_upipe_null_init, sktc_generic_upipe_fini,
	0x4000000100000000, 0xFFFFFFFF,
};

/****************************************************************/

static int
skt_writecfd_main(int argc, char *argv[])
{
	int error;
	ssize_t ret;
	channel_t channel;
	uuid_t channel_uuid;
	int channelfd;
	char buf[100];

	error = uuid_parse(argv[3], channel_uuid);
	SKTC_ASSERT_ERR(!error);

	channel = sktu_channel_create_extended(channel_uuid, 0,
	    CHANNEL_DIR_TX_RX, CHANNEL_RING_ID_ANY, NULL,
	    -1, -1, -1, -1, -1, -1, -1, 1, -1, -1);
	assert(channel);

	channelfd = os_channel_get_fd(channel);
	assert(channelfd != -1);

	ret = write(channelfd, buf, sizeof(buf));
	assert(ret == -1);
	SKTC_ASSERT_ERR(errno == EBADF);

	return 0;
}

struct skywalk_test skt_writecfd = {
	"writecfd", "test writing to channel fd",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_USER_PIPE,
	skt_writecfd_main, SKTC_GENERIC_UPIPE_ARGV,
	sktc_generic_upipe_null_init, sktc_generic_upipe_fini,
};


/****************************************************************/

static int
skt_readcfd_main(int argc, char *argv[])
{
	int error;
	ssize_t ret;
	channel_t channel;
	uuid_t channel_uuid;
	int channelfd;
	char buf[100];

	error = uuid_parse(argv[3], channel_uuid);
	SKTC_ASSERT_ERR(!error);

	channel = sktu_channel_create_extended(channel_uuid, 0,
	    CHANNEL_DIR_TX_RX, CHANNEL_RING_ID_ANY, NULL,
	    -1, -1, -1, -1, -1, -1, -1, 1, -1, -1);
	assert(channel);

	channelfd = os_channel_get_fd(channel);
	assert(channelfd != -1);

	ret = read(channelfd, buf, sizeof(buf));
	assert(ret == -1);
	SKTC_ASSERT_ERR(errno == EBADF);

	return 0;
}

struct skywalk_test skt_readcfd = {
	"readcfd", "test reading from channel fd",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_USER_PIPE,
	skt_readcfd_main, SKTC_GENERIC_UPIPE_ARGV,
	sktc_generic_upipe_null_init, sktc_generic_upipe_fini,
};

/****************************************************************/
