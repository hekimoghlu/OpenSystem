/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
 * @file sys/sysinfo.h
 * @brief System information.
 */

#include <sys/cdefs.h>
#include <linux/kernel.h>

__BEGIN_DECLS

/**
 * [sysinfo(2)](https://man7.org/linux/man-pages/man2/sysinfo.2.html) queries system information.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 */
int sysinfo(struct sysinfo* _Nonnull __info);

/**
 * [get_nprocs_conf(3)](https://man7.org/linux/man-pages/man3/get_nprocs_conf.3.html) returns
 * the total number of processors in the system.
 *
 * Available since API level 23.
 *
 * See also sysconf().
 */

#if __BIONIC_AVAILABILITY_GUARD(23)
int get_nprocs_conf(void) __INTRODUCED_IN(23);

/**
 * [get_nprocs(3)](https://man7.org/linux/man-pages/man3/get_nprocs.3.html) returns
 * the number of processors in the system that are currently on-line.
 *
 * Available since API level 23.
 *
 * See also sysconf().
 */
int get_nprocs(void) __INTRODUCED_IN(23);

/**
 * [get_phys_pages(3)](https://man7.org/linux/man-pages/man3/get_phys_pages.3.html) returns
 * the total number of physical pages in the system.
 *
 * Available since API level 23.
 *
 * See also sysconf().
 */
long get_phys_pages(void) __INTRODUCED_IN(23);

/**
 * [get_avphys_pages(3)](https://man7.org/linux/man-pages/man3/get_avphys_pages.3.html) returns
 * the number of physical pages in the system that are currently available.
 *
 * Available since API level 23.
 *
 * See also sysconf().
 */
long get_avphys_pages(void) __INTRODUCED_IN(23);
#endif /* __BIONIC_AVAILABILITY_GUARD(23) */


__END_DECLS
