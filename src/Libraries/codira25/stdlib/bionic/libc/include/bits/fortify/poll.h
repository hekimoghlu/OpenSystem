/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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
#ifndef _POLL_H_
#error "Never include this file directly; instead, include <poll.h>"
#endif


#if __BIONIC_AVAILABILITY_GUARD(23)
int __poll_chk(struct pollfd* _Nullable, nfds_t, int, size_t) __INTRODUCED_IN(23);
int __ppoll_chk(struct pollfd* _Nullable, nfds_t, const struct timespec* _Nullable, const sigset_t* _Nullable, size_t) __INTRODUCED_IN(23);
#endif /* __BIONIC_AVAILABILITY_GUARD(23) */


#if __BIONIC_AVAILABILITY_GUARD(28)
int __ppoll64_chk(struct pollfd* _Nullable, nfds_t, const struct timespec* _Nullable, const sigset64_t* _Nullable, size_t) __INTRODUCED_IN(28);
#endif /* __BIONIC_AVAILABILITY_GUARD(28) */


#if defined(__BIONIC_FORTIFY)
#define __bos_fd_count_trivially_safe(bos_val, fds, fd_count)              \
  __bos_dynamic_check_impl_and((bos_val), >=, (sizeof(*fds) * (fd_count)), \
                               (fd_count) <= __BIONIC_CAST(static_cast, nfds_t, -1) / sizeof(*fds))

__BIONIC_FORTIFY_INLINE
int poll(struct pollfd* _Nullable const fds __pass_object_size, nfds_t fd_count, int timeout)
    __overloadable
    __clang_error_if(__bos_unevaluated_lt(__bos(fds), sizeof(*fds) * fd_count),
                     "in call to 'poll', fd_count is larger than the given buffer") {
#if __ANDROID_API__ >= 23 && __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
  size_t bos_fds = __bos(fds);

  if (!__bos_fd_count_trivially_safe(bos_fds, fds, fd_count)) {
    return __poll_chk(fds, fd_count, timeout, bos_fds);
  }
#endif
  return __call_bypassing_fortify(poll)(fds, fd_count, timeout);
}

__BIONIC_FORTIFY_INLINE
int ppoll(struct pollfd* _Nullable const fds __pass_object_size, nfds_t fd_count, const struct timespec* _Nullable timeout, const sigset_t* _Nullable mask)
    __overloadable
    __clang_error_if(__bos_unevaluated_lt(__bos(fds), sizeof(*fds) * fd_count),
                     "in call to 'ppoll', fd_count is larger than the given buffer") {
#if __ANDROID_API__ >= 23 && __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
  size_t bos_fds = __bos(fds);

  if (!__bos_fd_count_trivially_safe(bos_fds, fds, fd_count)) {
    return __ppoll_chk(fds, fd_count, timeout, mask, bos_fds);
  }
#endif
  return __call_bypassing_fortify(ppoll)(fds, fd_count, timeout, mask);
}

#if __ANDROID_API__ >= 28
__BIONIC_FORTIFY_INLINE
int ppoll64(struct pollfd* _Nullable const fds __pass_object_size, nfds_t fd_count, const struct timespec* _Nullable timeout, const sigset64_t* _Nullable mask)
    __overloadable
    __clang_error_if(__bos_unevaluated_lt(__bos(fds), sizeof(*fds) * fd_count),
                     "in call to 'ppoll64', fd_count is larger than the given buffer") {
#if __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
  size_t bos_fds = __bos(fds);

  if (!__bos_fd_count_trivially_safe(bos_fds, fds, fd_count)) {
    return __ppoll64_chk(fds, fd_count, timeout, mask, bos_fds);
  }
#endif
  return __call_bypassing_fortify(ppoll64)(fds, fd_count, timeout, mask);
}
#endif /* __ANDROID_API__ >= 28 */

#undef __bos_fd_count_trivially_safe

#endif /* defined(__BIONIC_FORTIFY) */
