/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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
#include "krb5_locl.h"

KRB5_LIB_FUNCTION krb5_ssize_t KRB5_LIB_CALL
krb5_net_write (krb5_context context,
		void *p_fd,
		const void *buf,
		size_t len)
{
    krb5_socket_t fd = *((krb5_socket_t *)p_fd);
    return net_write(fd, buf, len);
}

KRB5_LIB_FUNCTION krb5_ssize_t KRB5_LIB_CALL
krb5_net_write_block(krb5_context context,
		     void *p_fd,
		     const void *buf,
		     size_t len,
		     time_t timeout)
{
  krb5_socket_t fd = *((krb5_socket_t *)p_fd);
  int ret;
  struct timeval tv, *tvp;
  const char *cbuf = (const char *)buf;
  size_t rem = len;
  ssize_t count;
  fd_set wfds;

  do {
      FD_ZERO(&wfds);
      FD_SET(fd, &wfds);

      if (timeout != 0) {
	  tv.tv_sec = timeout;
	  tv.tv_usec = 0;
	  tvp = &tv;
      } else
	  tvp = NULL;

      ret = select(fd + 1, NULL, &wfds, NULL, tvp);
      if (rk_IS_SOCKET_ERROR(ret)) {
	  if (rk_SOCK_ERRNO == EINTR)
	      continue;
	  return -1;
      }

#ifdef HAVE_WINSOCK
      if (ret == 0) {
	  WSASetLastError( WSAETIMEDOUT );
	  return 0;
      }

      count = send (fd, cbuf, rem, 0);

      if (rk_IS_SOCKET_ERROR(count)) {
	  return -1;
      }

#else
      if (ret == 0) {
	  return 0;
      }

      if (!FD_ISSET(fd, &wfds)) {
	  errno = ETIMEDOUT;
	  return -1;
      }

      count = write (fd, cbuf, rem);

      if (count < 0) {
	  if (errno == EINTR)
	      continue;
	  else
	      return count;
      }

#endif

      cbuf += count;
      rem -= count;

  } while (rem > 0);

  return len;
}
