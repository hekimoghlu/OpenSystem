/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <strings.h>
#include <sys/attr.h>
#include <sys/param.h>
#include <sys/mount.h>

static int
__statfs_ext_default(const char *path, int fd, struct statfs *buf)
{
	int ret = 0;

	if (path) {
		ret = statfs(path, buf);
	} else {
		ret = fstatfs(fd, buf);
	}

	return ret;
}

static int
__statfs_ext_noblock(const char *path, int fd, struct statfs *buf)
{
	int ret = 0;
	char *ptr;

	struct {
		uint32_t        size;
		attribute_set_t f_attrs;
		fsid_t          f_fsid;
		uint32_t        f_type;
		attrreference_t f_mntonname;
		uint32_t        f_flags;
		attrreference_t f_mntfromname;
		uint32_t        f_flags_ext;
		attrreference_t f_fstypename;
		uint32_t        f_fssubtype;
		uid_t           f_owner;
		char            f_mntonname_buf[MAXPATHLEN];
		char            f_mntfromname_buf[MAXPATHLEN];
		char            f_fstypename_buf[MFSTYPENAMELEN];
	} __attribute__((aligned(4), packed)) *attrbuf;

	struct attrlist al = {
		.bitmapcount = ATTR_BIT_MAP_COUNT,
		.commonattr = ATTR_CMN_FSID | ATTR_CMN_RETURNED_ATTRS,
		.volattr =  ATTR_VOL_INFO | ATTR_VOL_FSTYPE | ATTR_VOL_MOUNTPOINT |
	    ATTR_VOL_MOUNTFLAGS | ATTR_VOL_MOUNTEDDEVICE | ATTR_VOL_FSTYPENAME |
	    ATTR_VOL_FSSUBTYPE | ATTR_VOL_MOUNTEXTFLAGS | ATTR_VOL_OWNER,
	};

	attrbuf = malloc(sizeof(*attrbuf));
	if (attrbuf == NULL) {
		errno = ENOMEM;
		return -1;
	}
	bzero(attrbuf, sizeof(*attrbuf));

	if (path) {
		ret = getattrlist(path, &al, attrbuf, sizeof(*attrbuf), FSOPT_NOFOLLOW | FSOPT_RETURN_REALDEV);
	} else {
		ret = fgetattrlist(fd, &al, attrbuf, sizeof(*attrbuf), FSOPT_RETURN_REALDEV);
	}

	if (ret < 0) {
		goto out;
	}

	/* Update user structure */
	if (attrbuf->f_attrs.commonattr & ATTR_CMN_FSID) {
		buf->f_fsid = attrbuf->f_fsid;
	}
	if (attrbuf->f_attrs.volattr & ATTR_VOL_OWNER) {
		buf->f_owner = attrbuf->f_owner;
	}
	if (attrbuf->f_attrs.volattr & ATTR_VOL_FSTYPE) {
		buf->f_type = attrbuf->f_type;
	}
	if (attrbuf->f_attrs.volattr & ATTR_VOL_MOUNTFLAGS) {
		buf->f_flags = attrbuf->f_flags;
	}
	if (attrbuf->f_attrs.volattr & ATTR_VOL_FSSUBTYPE) {
		buf->f_fssubtype = attrbuf->f_fssubtype;
	}
	if (attrbuf->f_attrs.volattr & ATTR_VOL_FSTYPENAME) {
		ptr = (char *)&attrbuf->f_fstypename + attrbuf->f_fstypename.attr_dataoffset;
		strlcpy(buf->f_fstypename, ptr, sizeof(buf->f_fstypename));
	}
	if (attrbuf->f_attrs.volattr & ATTR_VOL_MOUNTPOINT) {
		ptr = (char *)&attrbuf->f_mntonname + attrbuf->f_mntonname.attr_dataoffset;
		strlcpy(buf->f_mntonname, ptr, sizeof(buf->f_mntonname));
	}
	if (attrbuf->f_attrs.volattr & ATTR_VOL_MOUNTEDDEVICE) {
		ptr = (char *)&attrbuf->f_mntfromname + attrbuf->f_mntfromname.attr_dataoffset;
		strlcpy(buf->f_mntfromname, ptr, sizeof(buf->f_mntfromname));
	}
	if (attrbuf->f_attrs.volattr & ATTR_VOL_MOUNTEXTFLAGS) {
		buf->f_flags_ext = attrbuf->f_flags_ext;
	}

out:
	free(attrbuf);
	return ret;
}

static int
__statfs_ext_impl(const char *path, int fd, struct statfs *buf, int flags)
{
	int ret = 0;

	bzero(buf, sizeof(struct statfs));

	/* Check for invalid flags */
	if (flags & ~(STATFS_EXT_NOBLOCK)) {
		errno = EINVAL;
		return -1;
	}

	/* Simply wrap statfs() or fstatfs() if no option is provided */
	if (flags == 0) {
		return __statfs_ext_default(path, fd, buf);
	}

	/* Retrieve filesystem statistics with extended options */
	if (flags & STATFS_EXT_NOBLOCK) {
		ret = __statfs_ext_noblock(path, fd, buf);
	}

	return ret;
}

int
fstatfs_ext(int fd, struct statfs *buf, int flags)
{
	/* fstatfs() sanity checks */
	if (fd < 0) {
		errno = EBADF;
		return -1;
	}
	if (buf == NULL) {
		errno = EFAULT;
		return -1;
	}

	return __statfs_ext_impl(NULL, fd, buf, flags);
}

int
statfs_ext(const char *path, struct statfs *buf, int flags)
{
	/* statfs() sanity checks */
	if (path == NULL) {
		errno = EFAULT;
		return -1;
	}
	if (buf == NULL) {
		errno = EFAULT;
		return -1;
	}

	return __statfs_ext_impl(path, -1, buf, flags);
}
