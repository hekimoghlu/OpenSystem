/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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
 * @file ifaddrs.h
 * @brief Access to network interface addresses.
 */

#include <sys/cdefs.h>
#include <netinet/in.h>
#include <sys/socket.h>

__BEGIN_DECLS

/**
 * Returned by getifaddrs() and freed by freeifaddrs().
 */
struct ifaddrs {
  /** Pointer to the next element in the linked list. */
  struct ifaddrs* _Nullable ifa_next;

  /** Interface name. */
  char* _Nullable ifa_name;
  /** Interface flags (like `SIOCGIFFLAGS`). */
  unsigned int ifa_flags;
  /** Interface address. */
  struct sockaddr* _Nullable ifa_addr;
  /** Interface netmask. */
  struct sockaddr* _Nullable ifa_netmask;

  union {
    /** Interface broadcast address (if IFF_BROADCAST is set). */
    struct sockaddr* _Nullable ifu_broadaddr;
    /** Interface destination address (if IFF_POINTOPOINT is set). */
    struct sockaddr* _Nullable ifu_dstaddr;
  } ifa_ifu;

  /** Unused. */
  void* _Nullable ifa_data;
};

/** Synonym for `ifa_ifu.ifu_broadaddr` in `struct ifaddrs`. */
#define ifa_broadaddr ifa_ifu.ifu_broadaddr
/** Synonym for `ifa_ifu.ifu_dstaddr` in `struct ifaddrs`. */
#define ifa_dstaddr ifa_ifu.ifu_dstaddr

/**
 * [getifaddrs(3)](https://man7.org/linux/man-pages/man3/getifaddrs.3.html) creates a linked list
 * of `struct ifaddrs`. The list must be freed by freeifaddrs().
 *
 * Returns 0 and stores the list in `*__list_ptr` on success,
 * and returns -1 and sets `errno` on failure.
 *
 * Available since API level 24.
 */

#if __BIONIC_AVAILABILITY_GUARD(24)
int getifaddrs(struct ifaddrs* _Nullable * _Nonnull __list_ptr) __INTRODUCED_IN(24);

/**
 * [freeifaddrs(3)](https://man7.org/linux/man-pages/man3/freeifaddrs.3.html) frees a linked list
 * of `struct ifaddrs` returned by getifaddrs().
 *
 * Available since API level 24.
 */
void freeifaddrs(struct ifaddrs* _Nullable __ptr) __INTRODUCED_IN(24);
#endif /* __BIONIC_AVAILABILITY_GUARD(24) */


__END_DECLS
