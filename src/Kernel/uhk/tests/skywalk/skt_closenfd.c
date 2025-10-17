/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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

static int
skt_closenfd_main(int argc, char *argv[])
{
	int error;
	nexus_controller_t ncd;
	int nfd;

	ncd = os_nexus_controller_create();
	assert(ncd);

	nfd = os_nexus_controller_get_fd(ncd);
	assert(nfd != -1);

	error = close(nfd); // expect guarded fd fail
	SKTC_ASSERT_ERR(!error);

	os_nexus_controller_destroy(ncd);

	return 1; // should not reach
}

struct skywalk_test skt_closenfd = {
	"closenfd", "test closing guarded nexus fd",
	SK_FEATURE_SKYWALK,
	skt_closenfd_main, { NULL }, NULL, NULL,
	0x4000000100000000, 0xFFFFFFFF,
};


/****************************************************************/

static int
skt_writenfd_main(int argc, char *argv[])
{
	nexus_controller_t ncd;
	int nfd;
	char buf[100] = { 0 };
	ssize_t ret;

	ncd = os_nexus_controller_create();
	assert(ncd);

	nfd = os_nexus_controller_get_fd(ncd);
	assert(nfd != -1);

	ret = write(nfd, buf, sizeof(buf));
	assert(ret == -1);
	assert(errno == EBADF);

	os_nexus_controller_destroy(ncd);

	return 0;
}

struct skywalk_test skt_writenfd = {
	"writenfd", "test writing to a guarded nexus fd",
	SK_FEATURE_SKYWALK,
	skt_writenfd_main, { NULL }, NULL, NULL, 0x9c00003, 0,
};

/****************************************************************/

static int
skt_readnfd_main(int argc, char *argv[])
{
	nexus_controller_t ncd;
	int nfd;
	char buf[100];
	ssize_t ret;

	ncd = os_nexus_controller_create();
	assert(ncd);

	nfd = os_nexus_controller_get_fd(ncd);
	assert(nfd != -1);

	ret = read(nfd, buf, sizeof(buf));
	assert(ret == -1);
	assert(errno == ENXIO);

	os_nexus_controller_destroy(ncd);

	return 0;
}

struct skywalk_test skt_readnfd = {
	"readnfd", "test reading from a guarded nexus fd",
	SK_FEATURE_SKYWALK,
	skt_readnfd_main, { NULL }, NULL, NULL,
};

/****************************************************************/
