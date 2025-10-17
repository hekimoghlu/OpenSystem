/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#include <unistd.h>
#include <errno.h>
#include <sys/event.h>
#include <sys/select.h>
#include <Block.h>
#include <darwintest.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.kevent"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("kevent"),
	T_META_RUN_CONCURRENTLY(true)
	);

#define TV(s) (struct timeval){ .tv_sec = s }

static void *
pthread_async_do(void *arg)
{
	void (^block)(void) = arg;
	block();
	Block_release(block);
	pthread_detach(pthread_self());
	return NULL;
}

static void
pthread_async(void (^block)(void))
{
	pthread_t th;
	int rc;

	rc = pthread_create(&th, NULL, pthread_async_do, Block_copy(block));
	T_QUIET; T_ASSERT_POSIX_SUCCESS(rc, "pthread_create");
}

T_DECL(kqueue_in_select, "make sure kqueue in select works", T_META_TAG_VM_PREFERRED)
{
	fd_set rd_set;
	int kq_fd, ret, nfd;
	struct kevent ret_kev;
	const struct kevent kev = {
		.ident = 1,
		.filter = EVFILT_USER,
		.flags = EV_ADD | EV_CLEAR,
	};

	T_ASSERT_POSIX_SUCCESS((kq_fd = kqueue()), NULL);
	ret = kevent(kq_fd, &kev, 1, NULL, 0, NULL);
	T_ASSERT_POSIX_SUCCESS(ret, "kevent");

	FD_ZERO(&rd_set);
	FD_SET(kq_fd, &rd_set);
	nfd = select(kq_fd + 1, &rd_set, NULL, NULL, &TV(1));
	T_EXPECT_EQ(nfd, 0, "no trigger");

	pthread_async(^{
		sleep(1);
		const struct kevent k = {
		        .ident = 1,
		        .filter = EVFILT_USER,
		        .flags = EV_ADD | EV_CLEAR,
		        .fflags = NOTE_TRIGGER,
		};

		T_ASSERT_POSIX_SUCCESS(kevent(kq_fd, &k, 1, NULL, 0, NULL), "trigger");
	});

	FD_ZERO(&rd_set);
	FD_SET(kq_fd, &rd_set);
	nfd = select(kq_fd + 1, &rd_set, NULL, NULL, &TV(5));
	T_EXPECT_EQ(nfd, 1, "kqueue triggered");

	FD_ZERO(&rd_set);
	FD_SET(kq_fd, &rd_set);
	nfd = select(kq_fd + 1, &rd_set, NULL, NULL, &TV(1));
	T_EXPECT_EQ(nfd, 1, "kqueue is still triggered");

	T_EXPECT_EQ(kevent(kq_fd, NULL, 0, &ret_kev, 1, NULL), 1, "pump event");

	FD_ZERO(&rd_set);
	FD_SET(kq_fd, &rd_set);
	nfd = select(kq_fd + 1, &rd_set, NULL, NULL, &TV(1));
	T_EXPECT_EQ(nfd, 0, "no trigger");
}
