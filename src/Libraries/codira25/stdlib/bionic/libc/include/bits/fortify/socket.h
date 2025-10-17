/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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
#ifndef _SYS_SOCKET_H_
#error "Never include this file directly; instead, include <sys/socket.h>"
#endif


#if __BIONIC_AVAILABILITY_GUARD(26)
ssize_t __sendto_chk(int, const void* _Nonnull, size_t, size_t, int, const struct sockaddr* _Nullable, socklen_t) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */

ssize_t __recvfrom_chk(int, void* _Nullable, size_t, size_t, int, struct sockaddr* _Nullable, socklen_t* _Nullable);

#if defined(__BIONIC_FORTIFY)

__BIONIC_FORTIFY_INLINE
ssize_t recvfrom(int fd, void* _Nullable const buf __pass_object_size0, size_t len, int flags, struct sockaddr* _Nullable src_addr, socklen_t* _Nullable addr_len)
    __overloadable
    __clang_error_if(__bos_unevaluated_lt(__bos0(buf), len),
                     "'recvfrom' called with size bigger than buffer") {
#if __ANDROID_API__ >= 24 && __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
  size_t bos = __bos0(buf);

  if (!__bos_trivially_ge(bos, len)) {
    return __recvfrom_chk(fd, buf, len, bos, flags, src_addr, addr_len);
  }
#endif
  return __call_bypassing_fortify(recvfrom)(fd, buf, len, flags, src_addr, addr_len);
}

__BIONIC_FORTIFY_INLINE
ssize_t sendto(int fd, const void* _Nonnull const buf __pass_object_size0, size_t len, int flags, const struct sockaddr* _Nullable dest_addr, socklen_t addr_len)
    __overloadable
    __clang_error_if(__bos_unevaluated_lt(__bos0(buf), len),
                     "'sendto' called with size bigger than buffer") {
#if __ANDROID_API__ >= 26 && __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
  size_t bos = __bos0(buf);

  if (!__bos_trivially_ge(bos, len)) {
    return __sendto_chk(fd, buf, len, bos, flags, dest_addr, addr_len);
  }
#endif
  return __call_bypassing_fortify(sendto)(fd, buf, len, flags, dest_addr, addr_len);
}

__BIONIC_FORTIFY_INLINE
ssize_t recv(int socket, void* _Nullable const buf __pass_object_size0, size_t len, int flags)
    __overloadable
    __clang_error_if(__bos_unevaluated_lt(__bos0(buf), len),
                     "'recv' called with size bigger than buffer") {
  return recvfrom(socket, buf, len, flags, NULL, 0);
}

__BIONIC_FORTIFY_INLINE
ssize_t send(int socket, const void* _Nonnull const buf __pass_object_size0, size_t len, int flags)
    __overloadable
    __clang_error_if(__bos_unevaluated_lt(__bos0(buf), len),
                     "'send' called with size bigger than buffer") {
  return sendto(socket, buf, len, flags, NULL, 0);
}

#endif /* __BIONIC_FORTIFY */
