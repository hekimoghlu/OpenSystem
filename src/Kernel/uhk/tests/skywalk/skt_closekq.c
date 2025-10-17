/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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
/* <rdar://problem/25158037> [N56 14A207] BTServer crash during BT off/on in  Watchdog_TimerSettings
 * verify that closing the kqueue fd causes select/poll/kevent to return
 */

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/event.h>
#include <darwintest.h>
#include "skywalk_test_driver.h"
#include "skywalk_test_common.h"

static int kq;

static void *
threadk(void *unused)
{
	struct kevent kev, refkev;
	int error;

	T_LOG("entering kevent in thread\n");

	memset(&kev, 0, sizeof(kev));
	refkev = kev;
	error = kevent(kq, NULL, 0, &kev, 1, NULL);
	SKTC_ASSERT_ERR(error == -1);
	SKTC_ASSERT_ERR(errno == EBADF || errno == EINTR);
	assert(!memcmp(&kev, &refkev, sizeof(refkev)));

	T_LOG("exiting thread\n");

	return NULL;
}

static int
skt_closekq_main_common(int argc, char *argv[], void * (*threadfunc)(void *unused))
{
	int error;
	pthread_t thread;

	kq = kqueue();
	assert(kq != -1);

	error = pthread_create(&thread, NULL, threadfunc, NULL);
	SKTC_ASSERT_ERR(!error);

	error = usleep(1000); // to make sure thread gets into select/poll/kevent
	SKTC_ASSERT_ERR(!error);

	T_LOG("closing kqueue in main\n");

	error = close(kq);
	SKTC_ASSERT_ERR(!error);

	T_LOG("joining thread in main\n");

	error = pthread_join(thread, NULL);
	SKTC_ASSERT_ERR(!error);

	T_LOG("exiting main\n");

	return 0;
}

static int
skt_closekqk_main(int argc, char *argv[])
{
	return skt_closekq_main_common(argc, argv, threadk);
}

struct skywalk_test skt_closekqk = {
	"closekqk", "test closing kqueue in kqueue",
	0, skt_closekqk_main, { NULL }, NULL, NULL,
};

/****************************************************************/
