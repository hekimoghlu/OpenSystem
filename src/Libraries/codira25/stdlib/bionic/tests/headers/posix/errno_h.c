/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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
// Copyright (C) 2017 The Android Open Source Project
// SPDX-License-Identifier: BSD-2-Clause

#include <errno.h>

#include "header_checks.h"

static void errno_h() {
  int error = errno;

  MACRO(E2BIG);
  MACRO(EACCES);
  MACRO(EADDRINUSE);
  MACRO(EADDRNOTAVAIL);
  MACRO(EAFNOSUPPORT);
  MACRO(EAGAIN);
  MACRO(EALREADY);
  MACRO(EBADF);
  MACRO(EBADMSG);
  MACRO(EBUSY);
  MACRO(ECANCELED);
  MACRO(ECHILD);
  MACRO(ECONNABORTED);
  MACRO(ECONNRESET);
  MACRO(EDEADLK);
  MACRO(EDESTADDRREQ);
  MACRO(EDOM);
  MACRO(EDQUOT);
  MACRO(EEXIST);
  MACRO(EFAULT);
  MACRO(EFBIG);
  MACRO(EHOSTUNREACH);
  MACRO(EIDRM);
  MACRO(EILSEQ);
  MACRO(EINPROGRESS);
  MACRO(EINTR);
  MACRO(EINVAL);
  MACRO(EIO);
  MACRO(EISCONN);
  MACRO(EISDIR);
  MACRO(ELOOP);
  MACRO(EMFILE);
  MACRO(EMLINK);
  MACRO(EMSGSIZE);
  MACRO(EMULTIHOP);
  MACRO(ENAMETOOLONG);
  MACRO(ENETDOWN);
  MACRO(ENETRESET);
  MACRO(ENETUNREACH);
  MACRO(ENFILE);
  MACRO(ENOBUFS);
  MACRO(ENODATA);
  MACRO(ENODEV);
  MACRO(ENOENT);
  MACRO(ENOEXEC);
  MACRO(ENOLCK);
  MACRO(ENOLINK);
  MACRO(ENOMEM);
  MACRO(ENOMSG);
  MACRO(ENOPROTOOPT);
  MACRO(ENOSPC);
  MACRO(ENOSR);
  MACRO(ENOSTR);
  MACRO(ENOSYS);
  MACRO(ENOTCONN);
  MACRO(ENOTDIR);
  MACRO(ENOTEMPTY);
  MACRO(ENOTRECOVERABLE);
  MACRO(ENOTSOCK);
  MACRO(ENOTSUP);
  MACRO(ENOTTY);
  MACRO(ENXIO);
  MACRO(EOPNOTSUPP);
  MACRO(EOVERFLOW);
  MACRO(EOWNERDEAD);
  MACRO(EPERM);
  MACRO(EPIPE);
  MACRO(EPROTO);
  MACRO(EPROTONOSUPPORT);
  MACRO(EPROTOTYPE);
  MACRO(ERANGE);
  MACRO(EROFS);
  MACRO(ESPIPE);
  MACRO(ESRCH);
  MACRO(ESTALE);
  MACRO(ETIME);
  MACRO(ETIMEDOUT);
  MACRO(ETXTBSY);
  MACRO(EWOULDBLOCK);
  MACRO(EXDEV);
}
