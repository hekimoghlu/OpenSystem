/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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

#include <stdatomic.h>
#include <sys/cdefs.h>

#include "platform/bionic/fdtrack.h"

#include "bionic/pthread_internal.h"
#include "private/ErrnoRestorer.h"
#include "private/bionic_tls.h"

extern "C" _Atomic(android_fdtrack_hook_t) __android_fdtrack_hook;
extern "C" bool __android_fdtrack_globally_disabled;

// Macro to record file descriptor creation.
// e.g.:
//   int socket(int domain, int type, int protocol) {
//     return FDTRACK_CREATE_NAME("socket", __socket(domain, type, protocol));
//   }
#define FDTRACK_CREATE_NAME(name, fd_value)                        \
  ({                                                               \
    int __fd = (fd_value);                                         \
    if (__fd != -1 && __predict_false(__android_fdtrack_hook) &&   \
        !__predict_false(__get_thread()->is_vforked())) {          \
      bionic_tls& tls = __get_bionic_tls();                        \
      /* fdtrack_disabled is only true during reentrant calls. */  \
      if (!__predict_false(tls.fdtrack_disabled) &&                \
          !__predict_false(__android_fdtrack_globally_disabled)) { \
        ErrnoRestorer r;                                           \
        tls.fdtrack_disabled = true;                               \
        android_fdtrack_event event;                               \
        event.fd = __fd;                                           \
        event.type = ANDROID_FDTRACK_EVENT_TYPE_CREATE;            \
        event.data.create.function_name = name;                    \
        atomic_load (&__android_fdtrack_hook)(&event);             \
        tls.fdtrack_disabled = false;                              \
      }                                                            \
    }                                                              \
    __fd;                                                          \
  })

// Macro to record file descriptor creation, with the current function's name.
// e.g.:
//   int socket(int domain, int type, int protocol) {
//     return FDTRACK_CREATE_NAME(__socket(domain, type, protocol));
//   }
#define FDTRACK_CREATE(fd_value) FDTRACK_CREATE_NAME(__func__, (fd_value))

// Macro to record file descriptor closure.
// Note that this does not actually close the file descriptor.
#define FDTRACK_CLOSE(fd_value)                                    \
  ({                                                               \
    int __fd = (fd_value);                                         \
    if (__fd != -1 && __predict_false(__android_fdtrack_hook) &&   \
        !__predict_false(__get_thread()->is_vforked())) {          \
      bionic_tls& tls = __get_bionic_tls();                        \
      if (!__predict_false(tls.fdtrack_disabled) &&                \
          !__predict_false(__android_fdtrack_globally_disabled)) { \
        int saved_errno = errno;                                   \
        tls.fdtrack_disabled = true;                               \
        android_fdtrack_event event;                               \
        event.fd = __fd;                                           \
        event.type = ANDROID_FDTRACK_EVENT_TYPE_CLOSE;             \
        atomic_load (&__android_fdtrack_hook)(&event);             \
        tls.fdtrack_disabled = false;                              \
        errno = saved_errno;                                       \
      }                                                            \
    }                                                              \
    __fd;                                                          \
  })
