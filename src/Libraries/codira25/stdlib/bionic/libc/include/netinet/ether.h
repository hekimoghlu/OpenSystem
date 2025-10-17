/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
 * @file netinet/ether.h
 * @brief Ethernet (MAC) addresses.
 */

#include <sys/cdefs.h>
#include <netinet/if_ether.h>

__BEGIN_DECLS

/**
 * [ether_ntoa(3)](https://man7.org/linux/man-pages/man3/ether_ntoa.3.html) returns a string
 * representation of the given Ethernet (MAC) address.
 *
 * Returns a pointer to a static buffer.
 */
char* _Nonnull ether_ntoa(const struct ether_addr* _Nonnull __addr);

/**
 * [ether_ntoa_r(3)](https://man7.org/linux/man-pages/man3/ether_ntoa_r.3.html) returns a string
 * representation of the given Ethernet (MAC) address.
 *
 * Returns a pointer to the given buffer.
 */
char* _Nonnull ether_ntoa_r(const struct ether_addr* _Nonnull __addr, char* _Nonnull __buf);

/**
 * [ether_aton(3)](https://man7.org/linux/man-pages/man3/ether_aton.3.html) returns an `ether_addr`
 * corresponding to the given Ethernet (MAC) address string.
 *
 * Returns a pointer to a static buffer, or NULL if the given string isn't a valid MAC address.
 */
struct ether_addr* _Nullable ether_aton(const char* _Nonnull __ascii);

/**
 * [ether_aton_r(3)](https://man7.org/linux/man-pages/man3/ether_aton_r.3.html) returns an
 * `ether_addr` corresponding to the given Ethernet (MAC) address string.
 *
 * Returns a pointer to the given buffer, or NULL if the given string isn't a valid MAC address.
 */
struct ether_addr* _Nullable ether_aton_r(const char* _Nonnull __ascii, struct ether_addr* _Nonnull __addr);

__END_DECLS
