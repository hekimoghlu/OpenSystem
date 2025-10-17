/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
#include <limits.h>
#include <sys/vfs.h>

static long __filesizebits(const struct statfs& s) {
  switch (s.f_type) {
    case JFFS2_SUPER_MAGIC:
    case MSDOS_SUPER_MAGIC:
    case NCP_SUPER_MAGIC:
      return 32;
  }
  // There won't be any new 32-bit file systems.
  return 64;
}

static long __link_max(const struct statfs& s) {
  // These constant values were taken from kernel headers.
  // They're not available in uapi headers.
  switch (s.f_type) {
    case EXT2_SUPER_MAGIC:
      return 32000;
    case MINIX_SUPER_MAGIC:
      return 250;
    case MINIX2_SUPER_MAGIC:
      return 65530;
    case REISERFS_SUPER_MAGIC:
      return 0xffff - 1000;
    case UFS_MAGIC:
      return 32000;
  }
  return LINK_MAX;
}

static long __2_symlinks(const struct statfs& s) {
  switch (s.f_type) {
    case ADFS_SUPER_MAGIC:
    case BFS_MAGIC:
    case CRAMFS_MAGIC:
    case EFS_SUPER_MAGIC:
    case MSDOS_SUPER_MAGIC:
    case QNX4_SUPER_MAGIC:
      return 0;
  }
  return 1;
}

static long __pathconf(const struct statfs& s, int name) {
  switch (name) {
    case _PC_FILESIZEBITS:
      return __filesizebits(s);

    case _PC_LINK_MAX:
      return __link_max(s);

    case _PC_MAX_CANON:
      return MAX_CANON;

    case _PC_MAX_INPUT:
      return MAX_INPUT;

    case _PC_NAME_MAX:
      return s.f_namelen;

    case _PC_PATH_MAX:
      return PATH_MAX;

    case _PC_PIPE_BUF:
      return PIPE_BUF;

    case _PC_2_SYMLINKS:
      return __2_symlinks(s);

    case _PC_ALLOC_SIZE_MIN:  /* fall through */
    case _PC_REC_XFER_ALIGN:
      return s.f_frsize;

    case _PC_REC_MIN_XFER_SIZE:
      return s.f_bsize;

    case _PC_CHOWN_RESTRICTED:
      return _POSIX_CHOWN_RESTRICTED;

    case _PC_NO_TRUNC:
      return _POSIX_NO_TRUNC;

    case _PC_VDISABLE:
      return _POSIX_VDISABLE;

    case _PC_ASYNC_IO:
    case _PC_PRIO_IO:
    case _PC_REC_INCR_XFER_SIZE:
    case _PC_REC_MAX_XFER_SIZE:
    case _PC_SYMLINK_MAX:
    case _PC_SYNC_IO:
      // No API to answer these: the caller will have to "try it and see".
      // This differs from the next case in not setting errno to EINVAL,
      // since we did understand the question --- we just don't have a
      // good answer.
      return -1;

    default:
      errno = EINVAL;
      return -1;
  }
}

long pathconf(const char* path, int name) {
  struct statfs sb;
  if (statfs(path, &sb) == -1) {
    return -1;
  }
  return __pathconf(sb, name);
}

long fpathconf(int fd, int name) {
  struct statfs sb;
  if (fstatfs(fd, &sb) == -1) {
    return -1;
  }
  return __pathconf(sb, name);
}
