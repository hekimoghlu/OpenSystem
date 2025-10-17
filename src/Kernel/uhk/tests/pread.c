/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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
#include <fcntl.h>
#include <unistd.h>
#include <sys/event.h>
#include <sys/mman.h>
#include <sys/socket.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.syscall.pread"),
	T_META_RUN_CONCURRENTLY(true)
	);

T_DECL(pread_regular_file,
    "test pread() on a regular file.") {
	char scratchfile_path[] = "/tmp/scratch.XXXXXX";
	int fd = mkstemp(scratchfile_path);
	T_ASSERT_POSIX_SUCCESS(fd, "created temporary file");
	T_ASSERT_POSIX_SUCCESS(unlink(scratchfile_path), "unlinked temporary file");

	char test_buffer[] = "a\0b";
	T_ASSERT_POSIX_SUCCESS(write(fd, test_buffer, 3), "wrote expected data");

	char pread_output_buffer[4];
	bzero(pread_output_buffer, 4);

	// Read one byte from zero.
	ssize_t result = pread(fd, pread_output_buffer, 1, 0);
	T_ASSERT_EQ(result, 1l, "pread 1 byte from 0");
	T_ASSERT_EQ(pread_output_buffer[0], 'a', "first byte output");
	T_ASSERT_EQ(pread_output_buffer[1], 0, "second byte output");
	T_ASSERT_EQ(pread_output_buffer[2], 0, "third byte output");
	T_ASSERT_EQ(pread_output_buffer[3], 0, "fourth byte output");

	// Read all bytes from zero.
	bzero(pread_output_buffer, 4);
	result = pread(fd, pread_output_buffer, 3, 0);
	T_ASSERT_EQ(result, 3l, "pread 3 bytes from 0");
	T_ASSERT_EQ(pread_output_buffer[0], 'a', "first byte output");
	T_ASSERT_EQ(pread_output_buffer[1], 0, "second byte output");
	T_ASSERT_EQ(pread_output_buffer[2], 'b', "third byte output");
	T_ASSERT_EQ(pread_output_buffer[3], 0, "fourth byte output");

	// Read more bytes than length from zero.
	bzero(pread_output_buffer, 4);
	result = pread(fd, pread_output_buffer, 4, 0);
	T_ASSERT_EQ(result, 3l, "pread 4 bytes from 0");
	T_ASSERT_EQ(pread_output_buffer[0], 'a', "first byte output");
	T_ASSERT_EQ(pread_output_buffer[1], 0, "second byte output");
	T_ASSERT_EQ(pread_output_buffer[2], 'b', "third byte output");
	T_ASSERT_EQ(pread_output_buffer[3], 0, "fourth byte output");

	// Read one byte from 2.
	bzero(pread_output_buffer, 4);
	result = pread(fd, pread_output_buffer, 1, 2);
	T_ASSERT_EQ(result, 1l, "pread 1 byte from 2");
	T_ASSERT_EQ(pread_output_buffer[0], 'b', "first byte output");
	T_ASSERT_EQ(pread_output_buffer[1], 0, "second byte output");
	T_ASSERT_EQ(pread_output_buffer[2], 0, "third byte output");
	T_ASSERT_EQ(pread_output_buffer[3], 0, "fourth byte output");

	// Read more bytes than length from 2.
	bzero(pread_output_buffer, 4);
	result = pread(fd, pread_output_buffer, 4, 2);
	T_ASSERT_EQ(result, 1l, "pread 4 bytes from 2");
	T_ASSERT_EQ(pread_output_buffer[0], 'b', "first byte output");
	T_ASSERT_EQ(pread_output_buffer[1], 0, "second byte output");
	T_ASSERT_EQ(pread_output_buffer[2], 0, "third byte output");
	T_ASSERT_EQ(pread_output_buffer[3], 0, "fourth byte output");
}

static void
test_pread_should_fail(int fd, int expected_errno)
{
	char output_buffer = 'A';
	ssize_t pread_result = pread(fd, &output_buffer, 1, 0);
	int err = errno;
	T_ASSERT_EQ(pread_result, (ssize_t)-1, "pread offset 0 size 1 returns -1");
	T_ASSERT_EQ(err, expected_errno, "errno is as expected");
	T_ASSERT_EQ(output_buffer, 'A', "input buffer is unchanged");

	pread_result = pread(fd, &output_buffer, 1, 1);
	err = errno;
	T_ASSERT_EQ(pread_result, (ssize_t)-1, "pread offset 1 size 1 returns -1");
	T_ASSERT_EQ(err, expected_errno, "errno is as expected");
	T_ASSERT_EQ(output_buffer, 'A', "input buffer is unchanged");

	pread_result = pread(fd, &output_buffer, 0, 0);
	err = errno;
	T_ASSERT_EQ(pread_result, (ssize_t)-1, "pread offset 0 size 0 returns -1");
	T_ASSERT_EQ(err, expected_errno, "errno is as expected");
	T_ASSERT_EQ(output_buffer, 'A', "input buffer is unchanged");

	pread_result = pread(fd, &output_buffer, 0, 1);
	err = errno;
	T_ASSERT_EQ(pread_result, (ssize_t)-1, "pread offset 1 size 0 returns -1");
	T_ASSERT_EQ(err, expected_errno, "errno is as expected");
	T_ASSERT_EQ(output_buffer, 'A', "input buffer is unchanged");
}

T_DECL(pread_socket,
    "test pread() on a socket.") {
	int sockets[2];
	int result = socketpair(AF_UNIX, SOCK_STREAM, 0, sockets);
	T_ASSERT_POSIX_SUCCESS(result, "Created socket pair");

	test_pread_should_fail(sockets[0], ESPIPE);
	test_pread_should_fail(sockets[1], ESPIPE);

	T_ASSERT_POSIX_SUCCESS(close(sockets[0]), "Closed socket 0");
	T_ASSERT_POSIX_SUCCESS(close(sockets[1]), "Closed socket 1");
}

T_DECL(pread_unix_shared_memory,
    "test pread() on unix shared memory.") {
	const char* memory_path = "test_pread_unix_shared_memory";
	int shm_fd = shm_open(memory_path, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
	T_ASSERT_POSIX_SUCCESS(shm_fd, "Created shared memory");

	test_pread_should_fail(shm_fd, ESPIPE);

	T_ASSERT_POSIX_SUCCESS(close(shm_fd), "Closed shm fd");
	T_ASSERT_POSIX_SUCCESS(shm_unlink(memory_path), "Unlinked");
}

T_DECL(pread_kqueue,
    "test pread() on kqueue.") {
	int queue = kqueue();
	T_ASSERT_POSIX_SUCCESS(queue, "Got kqueue");

	test_pread_should_fail(queue, ESPIPE);

	T_ASSERT_POSIX_SUCCESS(close(queue), "Closed queue");
}

T_DECL(pread_pipe,
    "test pread() on pipe.") {
	int pipe_fds[2];
	T_ASSERT_POSIX_SUCCESS(pipe(pipe_fds), "Created pipe");

	test_pread_should_fail(pipe_fds[0], ESPIPE);
	test_pread_should_fail(pipe_fds[1], EBADF);

	T_ASSERT_POSIX_SUCCESS(close(pipe_fds[1]), "Close write pipe");
	T_ASSERT_POSIX_SUCCESS(close(pipe_fds[0]), "Close read pipe");
}

T_DECL(pread_read_from_null,
    "test pread() from null.") {
	int fd = open("/dev/null", O_RDONLY);
	T_ASSERT_POSIX_SUCCESS(fd, "Opened /dev/null");

	char output_buffer = 'A';
	ssize_t pread_result = pread(fd, &output_buffer, 1, 0);
	T_ASSERT_EQ(pread_result, (ssize_t)0, "pread offset 0 size 1 returns 0");
	T_ASSERT_EQ(output_buffer, 'A', "input buffer is unchanged");

	pread_result = pread(fd, &output_buffer, 1, 1);
	T_ASSERT_EQ(pread_result, (ssize_t)0, "pread offset 1 size 1 returns 0");
	T_ASSERT_EQ(output_buffer, 'A', "input buffer is unchanged");

	pread_result = pread(fd, &output_buffer, 0, 0);
	T_ASSERT_EQ(pread_result, (ssize_t)0, "pread offset 0 size 0 returns 0");
	T_ASSERT_EQ(output_buffer, 'A', "input buffer is unchanged");

	pread_result = pread(fd, &output_buffer, 0, 1);
	T_ASSERT_EQ(pread_result, (ssize_t)0, "pread offset 1 size 0 returns 0");
	T_ASSERT_EQ(output_buffer, 'A', "input buffer is unchanged");

	T_ASSERT_POSIX_SUCCESS(close(fd), "Closed /dev/null");
}
