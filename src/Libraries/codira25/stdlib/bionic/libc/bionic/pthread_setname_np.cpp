/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
#include <pthread.h>

#include <fcntl.h>
#include <stdio.h> // For snprintf.
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "private/bionic_defs.h"
#include "private/ErrnoRestorer.h"
#include "pthread_internal.h"

// This value is not exported by kernel headers.
#define MAX_TASK_COMM_LEN 16

static int __open_task_comm_fd(pthread_t t, int flags, const char* caller) {
  char comm_name[64];
  snprintf(comm_name, sizeof(comm_name), "/proc/self/task/%d/comm",
           __pthread_internal_gettid(t, caller));
  return open(comm_name, O_CLOEXEC | flags);
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_getname_np(pthread_t t, char* buf, size_t buf_size) {
  ErrnoRestorer errno_restorer;

  if (buf_size < MAX_TASK_COMM_LEN) return ERANGE;

  // Getting our own name is an easy special case.
  if (t == pthread_self()) {
    return prctl(PR_GET_NAME, buf) ? errno : 0;
  }

  // We have to get another thread's name.
  int fd = __open_task_comm_fd(t, O_RDONLY, "pthread_getname_np");
  if (fd == -1) return errno;

  ssize_t n = TEMP_FAILURE_RETRY(read(fd, buf, buf_size));
  close(fd);

  if (n == -1) return errno;

  // The kernel adds a trailing '\n' to the /proc file,
  // so this is actually the normal case for short names.
  if (n > 0 && buf[n - 1] == '\n') {
    buf[n - 1] = '\0';
    return 0;
  }

  if (n == static_cast<ssize_t>(buf_size)) return ERANGE;
  buf[n] = '\0';
  return 0;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int pthread_setname_np(pthread_t t, const char* thread_name) {
  ErrnoRestorer errno_restorer;

  size_t thread_name_len = strlen(thread_name);
  if (thread_name_len >= MAX_TASK_COMM_LEN) return ERANGE;

  // Setting our own name is an easy special case.
  if (t == pthread_self()) {
    return prctl(PR_SET_NAME, thread_name) ? errno : 0;
  }

  // We have to set another thread's name.
  int fd = __open_task_comm_fd(t, O_WRONLY, "pthread_setname_np");
  if (fd == -1) return errno;

  ssize_t n = TEMP_FAILURE_RETRY(write(fd, thread_name, thread_name_len));
  close(fd);

  if (n == -1) return errno;
  if (n != static_cast<ssize_t>(thread_name_len)) return EIO;
  return 0;
}
