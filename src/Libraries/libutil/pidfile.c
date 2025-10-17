/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
#include <sys/cdefs.h>

#include <sys/param.h>
#include <sys/file.h>
#include <sys/stat.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <err.h>
#include <errno.h>
#include <libutil.h>

static int _pidfile_remove(struct pidfh *pfh, int freeit);

static int
pidfile_verify(struct pidfh *pfh)
{
	struct stat sb;

	if (pfh == NULL || pfh->pf_fd == -1)
	  return EINVAL;
	/*
	 * Check remembered descriptor.
	 */
	if (fstat(pfh->pf_fd, &sb) == -1)
		return (errno);
	if (sb.st_dev != pfh->pf_dev || sb.st_ino != pfh->pf_ino)
	  return EINVAL;
	return (0);
}

static int
pidfile_read(const char *path, pid_t *pidptr)
{
	char buf[16], *endptr;
	int error, fd, i;

	fd = open(path, O_RDONLY);
	if (fd == -1)
		return (errno);

	i = read(fd, buf, sizeof(buf) - 1);
	error = errno;	/* Remember errno in case close() wants to change it. */
	close(fd);
	if (i == -1)
		return (error);
	buf[i] = '\0';

	*pidptr = strtol(buf, &endptr, 10);
	if (endptr != &buf[i])
		return (EINVAL);

	return (0);
}

struct pidfh *
pidfile_open(const char *path, mode_t mode, pid_t *pidptr)
{
	struct pidfh *pfh;
	struct stat sb;
	int error, fd;

	pfh = malloc(sizeof(*pfh));
	if (pfh == NULL)
		return (NULL);

	if (path == NULL) {
		snprintf(pfh->pf_path, sizeof(pfh->pf_path), "/var/run/%s.pid",
		    getprogname());
	} else {
		strlcpy(pfh->pf_path, path, sizeof(pfh->pf_path));
	}
	if (strlen(pfh->pf_path) == sizeof(pfh->pf_path) - 1) {
		free(pfh);
		errno = ENAMETOOLONG;
		return (NULL);
	}

	/*
	 * Open the PID file and obtain exclusive lock.
	 * We truncate PID file here only to remove old PID immediatelly,
	 * PID file will be truncated again in pidfile_write(), so
	 * pidfile_write() can be called multiple times.
	 */
	fd = open(pfh->pf_path,
	    O_WRONLY | O_CREAT | O_EXLOCK | O_TRUNC | O_NONBLOCK, mode);
	if (fd == -1) {
		if (errno == EWOULDBLOCK && pidptr != NULL) {
			errno = pidfile_read(pfh->pf_path, pidptr);
			if (errno == 0)
				errno = EEXIST;
		}
		free(pfh);
		return (NULL);
	}
	/*
	 * Remember file information, so in pidfile_write() we are sure we write
	 * to the proper descriptor.
	 */
	if (fstat(fd, &sb) == -1) {
		error = errno;
		unlink(pfh->pf_path);
		close(fd);
		free(pfh);
		errno = error;
		return (NULL);
	}

	pfh->pf_fd = fd;
	pfh->pf_dev = sb.st_dev;
	pfh->pf_ino = sb.st_ino;

	return (pfh);
}

int
pidfile_write(struct pidfh *pfh)
{
	char pidstr[16];
	int error, fd;

	/*
	 * Check remembered descriptor, so we don't overwrite some other
	 * file if pidfile was closed and descriptor reused.
	 */
	errno = pidfile_verify(pfh);
	if (errno != 0) {
		/*
		 * Don't close descriptor, because we are not sure if it's ours.
		 */
		return (-1);
	}
	fd = pfh->pf_fd;

	/*
	 * Truncate PID file, so multiple calls of pidfile_write() are allowed.
	 */
	if (ftruncate(fd, 0) == -1) {
		error = errno;
		_pidfile_remove(pfh, 0);
		errno = error;
		return (-1);
	}

	snprintf(pidstr, sizeof(pidstr), "%u", getpid());
	if (pwrite(fd, pidstr, strlen(pidstr), 0) != (ssize_t)strlen(pidstr)) {
		error = errno;
		_pidfile_remove(pfh, 0);
		errno = error;
		return (-1);
	}

	return (0);
}

int
pidfile_close(struct pidfh *pfh)
{
	int error;

	error = pidfile_verify(pfh);
	if (error != 0) {
		errno = error;
		return (-1);
	}

	if (close(pfh->pf_fd) == -1)
		error = errno;
	free(pfh);
	if (error != 0) {
		errno = error;
		return (-1);
	}
	return (0);
}

static int
_pidfile_remove(struct pidfh *pfh, int freeit)
{
	int error;

	error = pidfile_verify(pfh);
	if (error != 0) {
		errno = error;
		return (-1);
	}

	if (unlink(pfh->pf_path) == -1)
		error = errno;
	if (flock(pfh->pf_fd, LOCK_UN) == -1) {
		if (error == 0)
			error = errno;
	}
	if (close(pfh->pf_fd) == -1) {
		if (error == 0)
			error = errno;
	}
	if (freeit)
		free(pfh);
	else
		pfh->pf_fd = -1;
	if (error != 0) {
		errno = error;
		return (-1);
	}
	return (0);
}

int
pidfile_remove(struct pidfh *pfh)
{

	return (_pidfile_remove(pfh, 1));
}
