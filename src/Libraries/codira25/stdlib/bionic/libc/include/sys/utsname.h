/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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

/**
 * @file sys/utsname.h
 * @brief The uname() function.
 */

#include <sys/cdefs.h>

__BEGIN_DECLS

/** The maximum length of any field in `struct utsname`. */
#define SYS_NMLN 65

/** The information returned by uname(). */
struct utsname {
  /** The OS name. "Linux" on Android. */
  char sysname[SYS_NMLN];
  /** The name on the network. Typically "localhost" on Android. */
  char nodename[SYS_NMLN];
  /** The OS release. Typically something like "4.4.115-g442ad7fba0d" on Android. */
  char release[SYS_NMLN];
  /** The OS version. Typically something like "#1 SMP PREEMPT" on Android. */
  char version[SYS_NMLN];
  /** The hardware architecture. Typically "aarch64" on Android. */
  char machine[SYS_NMLN];
  /** The domain name set by setdomainname(). Typically "localdomain" on Android. */
  char domainname[SYS_NMLN];
};

/**
 * [uname(2)](https://man7.org/linux/man-pages/man2/uname.2.html) returns information
 * about the kernel.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 */
int uname(struct utsname* _Nonnull __buf);

__END_DECLS
