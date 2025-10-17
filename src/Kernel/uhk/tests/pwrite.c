/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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
#include <stdlib.h>
#include <sys/event.h>
#include <sys/mman.h>
#include <sys/socket.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.syscall.pwrite"),
	T_META_RUN_CONCURRENTLY(true)
	);

static void
test_file_equals(int fd, const char* expected_buffer, size_t size)
{
	T_ASSERT_POSIX_SUCCESS(lseek(fd, 0, SEEK_SET), "Reset file offset");
	char* read_buffer = malloc(size);
	T_ASSERT_TRUE(read_buffer != 0, "Allocated read_buffer");
	T_ASSERT_EQ(read(fd, read_buffer, size), (ssize_t)size, "Read expected buffer");
	T_ASSERT_EQ(strncmp(read_buffer, expected_buffer, size), 0, "Buffer as expected");
	free(read_buffer);
}

T_DECL(pwrite_regular_file,
    "test pwrite() on a regular file.") {
	char scratchfile_path[] = "/tmp/scratch.XXXXXX";
	int fd = mkstemp(scratchfile_path);
	T_ASSERT_POSIX_SUCCESS(fd, "created temporary file");
	T_ASSERT_POSIX_SUCCESS(unlink(scratchfile_path), "unlinked temporary file");

	T_ASSERT_POSIX_SUCCESS(pwrite(fd, "a", 1, 1), "pwrite 'a' at offset 1");
	test_file_equals(fd, "\0a", 2);

	T_ASSERT_POSIX_SUCCESS(pwrite(fd, "bcd", 3, 0), "pwrite 'bcd' at offset 0");
	test_file_equals(fd, "bcd", 3);
}

static void
test_pwrite_should_fail(int fd, int expected_errno)
{
	char input_buffer = 'A';
	ssize_t pwrite_result = pwrite(fd, &input_buffer, 1, 0);
	int err = errno;
	T_ASSERT_EQ(pwrite_result, (ssize_t)-1, "pwrite offset 0 size 1 returns -1");
	T_ASSERT_EQ(err, expected_errno, "errno is as expected");
	T_ASSERT_EQ(input_buffer, 'A', "input buffer is unchanged");

	pwrite_result = pwrite(fd, &input_buffer, 1, 1);
	err = errno;
	T_ASSERT_EQ(pwrite_result, (ssize_t)-1, "pwrite offset 1 size 1 returns -1");
	T_ASSERT_EQ(err, expected_errno, "errno is as expected");
	T_ASSERT_EQ(input_buffer, 'A', "input buffer is unchanged");

	pwrite_result = pwrite(fd, &input_buffer, 0, 0);
	err = errno;
	T_ASSERT_EQ(pwrite_result, (ssize_t)-1, "pwrite offset 0 size 0 returns -1");
	T_ASSERT_EQ(err, expected_errno, "errno is as expected");
	T_ASSERT_EQ(input_buffer, 'A', "input buffer is unchanged");

	pwrite_result = pwrite(fd, &input_buffer, 0, 1);
	err = errno;
	T_ASSERT_EQ(pwrite_result, (ssize_t)-1, "pwrite offset 1 size 0 returns -1");
	T_ASSERT_EQ(err, expected_errno, "errno is as expected");
	T_ASSERT_EQ(input_buffer, 'A', "input buffer is unchanged");
}

T_DECL(pwrite_socket,
    "test pwrite() on a socket.") {
	int sockets[2];
	int result = socketpair(AF_UNIX, SOCK_STREAM, 0, sockets);
	T_ASSERT_POSIX_SUCCESS(result, "Created socket pair");

	test_pwrite_should_fail(sockets[0], ESPIPE);
	test_pwrite_should_fail(sockets[1], ESPIPE);

	T_ASSERT_POSIX_SUCCESS(close(sockets[0]), "Closed socket 0");
	T_ASSERT_POSIX_SUCCESS(close(sockets[1]), "Closed socket 1");
}

T_DECL(pwrite_unix_shared_memory,
    "test pwrite() on unix shared memory.") {
	const char* memory_path = "test_pwrite_unix_shared_memory";
	int shm_fd = shm_open(memory_path, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
	T_ASSERT_POSIX_SUCCESS(shm_fd, "Created shared memory");

	test_pwrite_should_fail(shm_fd, ESPIPE);

	T_ASSERT_POSIX_SUCCESS(close(shm_fd), "Closed shm fd");
	T_ASSERT_POSIX_SUCCESS(shm_unlink(memory_path), "Unlinked");
}

T_DECL(pwrite_kqueue,
    "test pwrite() on kqueue.") {
	int queue = kqueue();
	T_ASSERT_POSIX_SUCCESS(queue, "Got kqueue");

	test_pwrite_should_fail(queue, ESPIPE);

	T_ASSERT_POSIX_SUCCESS(close(queue), "Close queue");
}

T_DECL(pwrite_pipe,
    "test pwrite() on pipe.") {
	int pipe_fds[2];
	T_ASSERT_POSIX_SUCCESS(pipe(pipe_fds), "Created pipe");

	test_pwrite_should_fail(pipe_fds[0], EBADF);
	test_pwrite_should_fail(pipe_fds[1], ESPIPE);

	T_ASSERT_POSIX_SUCCESS(close(pipe_fds[1]), "Close write pipe");
	T_ASSERT_POSIX_SUCCESS(close(pipe_fds[0]), "Close read pipe");
}

T_DECL(pwrite_write_to_dev_null,
    "test pwrite() from null.") {
	int fd = open("/dev/null", O_RDONLY);
	T_ASSERT_POSIX_SUCCESS(fd, "Opened /dev/null");

	char buffer = 'A';
	errno = 0;
	ssize_t pwrite_result = pwrite(fd, &buffer, 1, 0);
	int err = errno;
	T_ASSERT_EQ(pwrite_result, (ssize_t)-1, "pwrite offset 0 size 1 returns -1");
	T_ASSERT_EQ(err, EBADF, "pwrite on /dev/null results in EBADF");

	errno = 0;
	pwrite_result = pwrite(fd, &buffer, 1, 1);
	err = errno;
	T_ASSERT_EQ(pwrite_result, (ssize_t)-1, "pwrite offset 1 size 1 returns -1");
	T_ASSERT_EQ(buffer, 'A', "input buffer is unchanged");
	T_ASSERT_EQ(err, EBADF, "pwrite on /dev/null results in EBADF");

	errno = 0;
	pwrite_result = pwrite(fd, &buffer, 0, 0);
	err = errno;
	T_ASSERT_EQ(pwrite_result, (ssize_t)-1, "pwrite offset 0 size 0 returns -1");
	T_ASSERT_EQ(err, EBADF, "pwrite on /dev/null results in EBADF");
	T_ASSERT_EQ(buffer, 'A', "input buffer is unchanged");

	errno = 0;
	pwrite_result = pwrite(fd, &buffer, 0, 1);
	err = errno;
	T_ASSERT_EQ(pwrite_result, (ssize_t)-1, "pwrite offset 1 size 0 returns -1");
	T_ASSERT_EQ(err, EBADF, "pwrite on /dev/null results in EBADF");
	T_ASSERT_EQ(buffer, 'A', "input buffer is unchanged");

	T_ASSERT_POSIX_SUCCESS(close(fd), "Closed /dev/null");
}
