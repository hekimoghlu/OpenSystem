/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
#include <darwintest_multiprocess.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <pthread.h>
#include <stdlib.h>
#include <signal.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>

T_GLOBAL_META(T_META_RADAR_COMPONENT_NAME("xnu"),
    T_META_RADAR_COMPONENT_VERSION("kevent"));

T_DECL(pipe_noblock_kevent,
    "Set a pipe and no block and setup EVFLT_WRITE kevent on it and make sure it does not fire when the pipe is full")
{
	int fd[2], write_fd;
	dispatch_queue_t dq1 = dispatch_queue_create("com.apple.test.pipe_noblock_kevent.queue", DISPATCH_QUEUE_SERIAL);

	pipe(fd);
	write_fd = fd[1];
	__block int iter = 1;

	/* Make sure the pipe is No block */
	fcntl(write_fd, F_SETFL, (O_NONBLOCK));

	dispatch_source_t write_source = dispatch_source_create(DISPATCH_SOURCE_TYPE_WRITE, (uintptr_t)write_fd, 0, dq1);
	dispatch_source_set_event_handler(write_source, ^{
		unsigned long length = dispatch_source_get_data(write_source);

		T_LOG("Iteration: %d, Length available: %lu\n", iter++, length);

		char buf[512] = "deadbeef";
		ssize_t rv = write(write_fd, buf, 512);
		T_EXPECT_POSIX_SUCCESS(rv, "write success");
		if (rv < 0) {
		        T_FAIL("Write should have succeeded but failed with error %ld", rv);
		        T_END;
		}
	});

	dispatch_resume(write_source);

	T_LOG("Arming a timer for 15 seconds to exit, assuming kevent will block before that");
	dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 15 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
		T_LOG("PASS: Kevent blocked as expected in the EVFLT_WRITE");
		T_END;
	});

	dispatch_main();
}
