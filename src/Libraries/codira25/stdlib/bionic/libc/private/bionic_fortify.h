/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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
#pragma once

#include <poll.h> // For struct pollfd.
#include <stdarg.h>
#include <stdlib.h>
#include <sys/select.h> // For struct fd_set.

#include <async_safe/log.h>

//
// LLVM can't inline variadic functions, and we don't want one definition of
// this per #include in libc.so, so no `static`.
//
inline __noreturn __printflike(1, 2) void __fortify_fatal(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  async_safe_fatal_va_list("FORTIFY", fmt, args);
  va_end(args);
  abort();
}

//
// Common helpers.
//

static inline void __check_fd_set(const char* fn, int fd, size_t set_size) {
  if (__predict_false(fd < 0)) {
    __fortify_fatal("%s: file descriptor %d < 0", fn, fd);
  }
  if (__predict_false(fd >= FD_SETSIZE)) {
    __fortify_fatal("%s: file descriptor %d >= FD_SETSIZE %d", fn, fd, FD_SETSIZE);
  }
  if (__predict_false(set_size < sizeof(fd_set))) {
    __fortify_fatal("%s: set size %zu is too small to be an fd_set", fn, set_size);
  }
}

static inline void __check_pollfd_array(const char* fn, size_t fds_size, nfds_t fd_count) {
  size_t pollfd_array_length = fds_size / sizeof(pollfd);
  if (__predict_false(pollfd_array_length < fd_count)) {
    __fortify_fatal("%s: %zu-element pollfd array too small for %u fds",
                    fn, pollfd_array_length, fd_count);
  }
}

static inline void __check_count(const char* fn, const char* identifier, size_t value) {
  if (__predict_false(value > SSIZE_MAX)) {
    __fortify_fatal("%s: %s %zu > SSIZE_MAX", fn, identifier, value);
  }
}

static inline void __check_buffer_access(const char* fn, const char* action,
                                         size_t claim, size_t actual) {
  if (__predict_false(claim > actual)) {
    __fortify_fatal("%s: prevented %zu-byte %s %zu-byte buffer", fn, claim, action, actual);
  }
}
