/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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
#include <os/assumes.h>
#include <os/once_private.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

/*
 * posix_openpt call for cloning pty implementation.
 *
 * Just open /dev/ptmx
 */
int
posix_openpt(int flags)
{
	int fd = open("/dev/ptmx", flags);
	if (fd >= 0)
		return fd;
	return -1;
}

/*
 * grantpt call for cloning pty implementation.
 *
 * Change UID and GID of slave pty associated with master pty whose
 * fd is provided, to the real UID and real GID of the calling thread.
 */
int
grantpt(int fd)
{
	return ioctl(fd, TIOCPTYGRANT);
}

// defined by TIOCPTYGNAME
#define PTSNAME_MAX_SIZE 128

static pthread_key_t ptsname_buffer_specific_key;
static os_once_t ptsname_once;

static void
ptsname_once_init(void *ctx __unused)
{
	int ret = pthread_key_create(&ptsname_buffer_specific_key, free);
	os_assert_zero(ret);
}

/*
 * ptsname call for cloning pty implementation.
 *
 * NOTE: Returns a pointer to a static buffer, which will be overwritten on
 * subsequent calls.
 */
char *
ptsname(int fd)
{
	os_once(&ptsname_once, NULL, ptsname_once_init);
	char *ptsnamebuf = pthread_getspecific(ptsname_buffer_specific_key);

	if (ptsnamebuf == NULL) {
		ptsnamebuf = malloc(PTSNAME_MAX_SIZE);
		os_assert(ptsnamebuf);

		int error = pthread_setspecific(ptsname_buffer_specific_key, ptsnamebuf);
		os_assert_zero(error);
	}

	int error = ptsname_r(fd, ptsnamebuf, PTSNAME_MAX_SIZE);

	return error ? NULL : ptsnamebuf;
}

int
ptsname_r(int fd, char *buffer, size_t buflen)
{
	int error;
	struct stat sbuf;
	char ptsnamebuf[PTSNAME_MAX_SIZE];

	if (!buffer) {
		errno = EINVAL;
		return -1;
	}

	error = ioctl(fd, TIOCPTYGNAME, ptsnamebuf);
	if (error) {
		return -1;
	}

	/*
	 * XXX TSD
	 *
	 * POSIX: Handle device rename test case, which is expected
	 * to fail if the pty has been renamed.
	 */
	error = stat(ptsnamebuf, &sbuf);
	if (error) {
		return -1;
	}

	size_t len = strlen(ptsnamebuf) + 1;
	if (buflen < len) {
		errno = ERANGE;
		return -1;
	}

	memcpy(buffer, ptsnamebuf, len);

	return 0;
}

/*
 * unlockpt call for cloning pty implementation.
 *
 * Unlock the slave pty associated with the master to which fd refers.
 */
int
unlockpt(int fd)
{
	return ioctl(fd, TIOCPTYUNLK);
}
