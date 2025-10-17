/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 30, 2021.
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
/* XXX temporary implementation using __acl__ file */

#include <sys/appleapiopts.h>
#include <sys/types.h>
#include <sys/acl.h>
#include <sys/stat.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "aclvar.h"

static acl_t	acl_get_file1(const char *path, acl_type_t acl_type, int follow);

int acl_delete_fd_np(int filedes, acl_type_t type);
int
acl_delete_fd_np(int filedes, acl_type_t type)
{
	errno = ENOTSUP;
	return(-1);
}

int acl_delete_file_np(const char *path, acl_type_t type);
int
acl_delete_file_np(const char *path, acl_type_t type)
{
	errno = ENOTSUP;
	return(-1);
}

int acl_delete_link_np(const char *path, acl_type_t type);
int
acl_delete_link_np(const char *path, acl_type_t type)
{
	errno = ENOTSUP;
	return(-1);
}

acl_t
acl_get_fd(int fd)
{
	return(acl_get_fd_np(fd, ACL_TYPE_EXTENDED));
}

acl_t
acl_get_fd_np(int fd, acl_type_t type)
{
	filesec_t	fsec;
	acl_t		acl;
	struct stat	sb;

	if (type != ACL_TYPE_EXTENDED) {
		errno = EINVAL;
		return(NULL);
	}
	if ((fsec = filesec_init()) == NULL)
		return(NULL);

	acl = NULL;
	if (fstatx_np(fd, &sb, fsec) == 0)
		filesec_get_property(fsec, FILESEC_ACL, &acl);
	filesec_free(fsec);
	return(acl);
}

static acl_t
acl_get_file1(const char *path, acl_type_t acl_type, int follow)
{
	filesec_t	fsec;
	acl_t		acl;
	struct stat	sb;

	if (acl_type != ACL_TYPE_EXTENDED) {
		errno = EINVAL;
		return(NULL);
	}
	if ((fsec = filesec_init()) == NULL)
		return(NULL);

	acl = NULL;
	if ((follow ? statx_np(path, &sb, fsec) : lstatx_np(path, &sb, fsec)) == 0)
		filesec_get_property(fsec, FILESEC_ACL, &acl);
	filesec_free(fsec);
	return(acl);
}

acl_t
acl_get_file(const char *path, acl_type_t type)
{
	return(acl_get_file1(path, type, 1 /* follow */));
}

acl_t
acl_get_link_np(const char *path, acl_type_t type)
{
	return(acl_get_file1(path, type, 0 /* no follow */));
}

int
acl_set_fd_np(int fd, acl_t acl, acl_type_t type)
{
	filesec_t	fsec;
	int		error;

	if ((fsec = filesec_init()) == NULL)
		return(-1);
	if ((filesec_set_property(fsec, FILESEC_ACL, &acl)) != 0) {
		filesec_free(fsec);
		return(-1);
	}
	error = fchmodx_np(fd, fsec);
	filesec_free(fsec);
	return((error == 0) ? 0 : -1);
}

int
acl_set_fd(int fd, acl_t acl)
{
	return(acl_set_fd_np(fd, acl, ACL_TYPE_EXTENDED));
}

int
acl_set_file(const char *path, acl_type_t acl_type, acl_t acl)
{
	filesec_t	fsec;
	int		error;

	if ((fsec = filesec_init()) == NULL)
		return(-1);
	if (filesec_set_property(fsec, FILESEC_ACL, &acl) != 0) {
		filesec_free(fsec);
		return(-1);
	}
	error = chmodx_np(path, fsec);
	filesec_free(fsec);
	return((error == 0) ? 0 : -1);
}

int
acl_set_link_np(const char *path, acl_type_t acl_type, acl_t acl)
{
	struct stat s;

	if(lstat(path, &s) < 0)
		return(-1);
	if(S_ISLNK(s.st_mode)) {
		errno = ENOTSUP;
		return(-1);
	}
	return(acl_set_file(path, acl_type, acl));
}

/*
 * Not applicable; not supported.
 */
int
acl_delete_def_file(__unused const char *path)
{
	errno = ENOTSUP;
	return(-1);
}
