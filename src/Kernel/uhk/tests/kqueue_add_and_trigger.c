/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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
#include <darwintest.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

/* <rdar://problem/28139044> EVFILT_USER doesn't properly support add&fire atomic combination
 *
 * Chek that using EV_ADD and EV_TRIGGER on a EV_USER actually trigger the event just added.
 *
 */

T_DECL(kqueue_add_and_trigger_evfilt_user, "Add and trigger EVFILT_USER events with kevent ", T_META_TAG_VM_PREFERRED)
{
	int kq_fd, ret;
	struct kevent ret_kev;
	const struct kevent kev = {
		.ident = 1,
		.filter = EVFILT_USER,
		.flags = EV_ADD | EV_CLEAR,
		.fflags = NOTE_TRIGGER,
	};
	const struct timespec timeout = {
		.tv_sec = 1,
		.tv_nsec = 0,
	};

	T_ASSERT_POSIX_SUCCESS((kq_fd = kqueue()), NULL);
	ret = kevent(kq_fd, &kev, 1, &ret_kev, 1, &timeout);

	T_ASSERT_POSIX_SUCCESS(ret, "kevent");

	T_ASSERT_EQ(ret, 1, "kevent with add and trigger, ret");
	T_ASSERT_EQ(ret_kev.ident, 1, "kevent with add and trigger, ident");
	T_ASSERT_EQ(ret_kev.filter, EVFILT_USER, "kevent with add and trigger, filter");
}
