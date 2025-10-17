/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 29, 2022.
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
#include <darwintest.h>
#include <darwintest_utils.h>
#include <dispatch/dispatch.h>
#include <sys/guarded.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.fd"),
	T_META_RUN_CONCURRENTLY(true));

T_DECL(fd_guard_monitored, "Test that we can guard fds in kevent", T_META_TAG_VM_PREFERRED)
{
	static int pfd[2];
	static dispatch_source_t ds;
	guardid_t guard = (uintptr_t)&pfd;

	T_ASSERT_POSIX_SUCCESS(pipe(pfd), "pipe");

	ds = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ,
	    (uintptr_t)pfd[0], 0, NULL);
	dispatch_source_set_event_handler(ds, ^{ });
	dispatch_activate(ds);

	T_EXPECT_POSIX_SUCCESS(change_fdguard_np(pfd[0], NULL, 0,
	    &guard, GUARD_DUP | GUARD_CLOSE, NULL), "change_fdguard_np");
}
