/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
#ifndef __DARWIN_TAPI_H
#define __DARWIN_TAPI_H

#if !DARWIN_TAPI
#error "This header is for the installapi action only"
#endif

#include <os/base.h>
#include <os/availability.h>
#include <mach/kern_return.h>
#include <mach/port.h>
#include <mach/mach_port.h>

#undef os_assert_mach
#undef os_assert_mach_port_status

// Duplicate declarations to make TAPI happy.
API_AVAILABLE(macos(10.14), ios(12.0), tvos(12.0), watchos(5.0))
OS_EXPORT OS_NONNULL1
void
os_assert_mach(const char *op, kern_return_t kr);

API_AVAILABLE(macos(10.14), ios(12.0), tvos(12.0), watchos(5.0))
OS_EXPORT
void
os_assert_mach_port_status(const char *desc, mach_port_t p,
		mach_port_status_t *expected);

// TAPI and the compiler don't agree about header search paths, so if TAPI found
// our header in the SDK, and we've increased the API version, help it out.
#if DARWIN_API_VERSION < 20170407
#define DARWIN_API_AVAILABLE_20170407
#endif

#if DARWIN_API_VERSION < 20180727
#define DARWIN_API_AVAILABLE_20180727
#endif

#if DARWIN_API_VERSION < 20181020
#define DARWIN_API_AVAILABLE_20181020
#endif

#if DARWIN_API_VERSION < 20190830
#define DARWIN_API_AVAILABLE_20190830
#endif

#if DARWIN_API_VERSION < 20191015
#define DARWIN_API_AVAILABLE_20191015
#endif

#if !defined(LINKER_SET_ENTRY)
#define LINKER_SET_ENTRY(_x, _y)
#endif

#endif // __DARWIN_TAPI_H
