/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#include <dirent.h>

#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <android/fdsan.h>

#include "private/bionic_fortify.h"
#include "private/ErrnoRestorer.h"
#include "private/ScopedPthreadMutexLocker.h"

extern "C" int __getdents64(unsigned int, dirent*, unsigned int);

// Apportable decided to copy the data structure from this file
// and use it in their own code, but they also call into readdir.
// In order to avoid a lockup, the structure must be maintained in
// the exact same order as in L and below. New structure members
// need to be added to the end of this structure.
// See b/21037208 for more details.
struct DIR {
  int fd_;
  size_t available_bytes_;
  dirent* next_;
  pthread_mutex_t mutex_;
  dirent buff_[15];
  long current_pos_;
};

#define CHECK_DIR(d) if (d == nullptr) __fortify_fatal("%s: null DIR*", __FUNCTION__)

static uint64_t __get_dir_tag(DIR* dir) {
  return android_fdsan_create_owner_tag(ANDROID_FDSAN_OWNER_TYPE_DIR,
                                        reinterpret_cast<uint64_t>(dir));
}

static DIR* __allocate_DIR(int fd) {
  DIR* d = reinterpret_cast<DIR*>(malloc(sizeof(DIR)));
  if (d == nullptr) {
    return nullptr;
  }
  d->fd_ = fd;
  android_fdsan_exchange_owner_tag(fd, 0, __get_dir_tag(d));
  d->available_bytes_ = 0;
  d->next_ = nullptr;
  d->current_pos_ = 0L;
  pthread_mutex_init(&d->mutex_, nullptr);
  return d;
}

int dirfd(DIR* d) {
  CHECK_DIR(d);
  return d->fd_;
}

DIR* fdopendir(int fd) {
  // Is 'fd' actually a directory?
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    return nullptr;
  }
  if (!S_ISDIR(sb.st_mode)) {
    errno = ENOTDIR;
    return nullptr;
  }

  return __allocate_DIR(fd);
}

DIR* opendir(const char* path) {
  int fd = open(path, O_CLOEXEC | O_DIRECTORY | O_RDONLY);
  return (fd != -1) ? __allocate_DIR(fd) : nullptr;
}

static bool __fill_DIR(DIR* d) {
  CHECK_DIR(d);
  int rc = TEMP_FAILURE_RETRY(__getdents64(d->fd_, d->buff_, sizeof(d->buff_)));
  if (rc <= 0) {
    return false;
  }
  d->available_bytes_ = rc;
  d->next_ = d->buff_;
  return true;
}

static dirent* __readdir_locked(DIR* d) {
  if (d->available_bytes_ == 0 && !__fill_DIR(d)) {
    return nullptr;
  }

  dirent* entry = d->next_;
  d->next_ = reinterpret_cast<dirent*>(reinterpret_cast<char*>(entry) + entry->d_reclen);
  d->available_bytes_ -= entry->d_reclen;
  // The directory entry offset uses 0, 1, 2 instead of real file offset,
  // so the value range of long type is enough.
  d->current_pos_ = static_cast<long>(entry->d_off);
  return entry;
}

dirent* readdir(DIR* d) {
  CHECK_DIR(d);
  ScopedPthreadMutexLocker locker(&d->mutex_);
  return __readdir_locked(d);
}
__strong_alias(readdir64, readdir);

int readdir_r(DIR* d, dirent* entry, dirent** result) {
  CHECK_DIR(d);

  ErrnoRestorer errno_restorer;

  *result = nullptr;
  errno = 0;

  ScopedPthreadMutexLocker locker(&d->mutex_);

  dirent* next = __readdir_locked(d);
  if (errno != 0 && next == nullptr) {
    return errno;
  }

  if (next != nullptr) {
    memcpy(entry, next, next->d_reclen);
    *result = entry;
  }
  return 0;
}
__strong_alias(readdir64_r, readdir_r);

int closedir(DIR* d) {
  if (d == nullptr) {
    errno = EINVAL;
    return -1;
  }

  int fd = d->fd_;
  pthread_mutex_destroy(&d->mutex_);
  int rc = android_fdsan_close_with_tag(fd, __get_dir_tag(d));
  free(d);
  return rc;
}

void rewinddir(DIR* d) {
  CHECK_DIR(d);

  ScopedPthreadMutexLocker locker(&d->mutex_);
  lseek(d->fd_, 0, SEEK_SET);
  d->available_bytes_ = 0;
  d->current_pos_ = 0L;
}

void seekdir(DIR* d, long offset) {
  CHECK_DIR(d);

  ScopedPthreadMutexLocker locker(&d->mutex_);
  off_t ret = lseek(d->fd_, offset, SEEK_SET);
  if (ret != -1L) {
    d->available_bytes_ = 0;
    d->current_pos_ = ret;
  }
}

long telldir(DIR* d) {
  CHECK_DIR(d);

  return d->current_pos_;
}

int alphasort(const dirent** a, const dirent** b) {
  return strcoll((*a)->d_name, (*b)->d_name);
}
__strong_alias(alphasort64, alphasort);
