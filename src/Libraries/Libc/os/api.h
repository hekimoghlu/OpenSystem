/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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
/*!
 * @header
 * API macros. libdarwin provides APIs for low-level userspace projects in the
 * Darwin operating system.
 *
 * - C language additions
 * - POSIX and BSD API additions
 * - POSIX and BSD convenience wrappers
 * - Mach API additions and wrappers with clearer semantics
 *
 * Additions which extend the C language are not prefixed and are therefore not
 * included by default when including this header.
 *
 * Additions to API families conforming to ANSI C carry the "os_" prefix.
 *
 * Additions to API families conforming to POSIX carry the "_np" ("Not POSIX")
 * suffix.
 *
 * Additions to API families conforming to both POSIX and ANSI C carry the "_np"
 * suffix.
 *
 * Convenience wrappers for POSIX and BSD APIs carry the "os_" prefix.
 *
 * New APIs formalizing Darwin workflows carry the "os_" prefix.
 */
#ifndef __DARWIN_API_H
#define __DARWIN_API_H

#include <os/availability.h>
#include <stdint.h>

OS_ASSUME_PTR_ABI_SINGLE_BEGIN;

/*!
 * @const DARWIN_API_VERSION
 * The API version of the library. This version will be changed in accordance
 * with new API introductions so that callers may submit code to the build that
 * adopts those new APIs before the APIs land by using the following pattern:
 *
 * #if DARWIN_API_VERSION >= 20180424
 * darwin_new_api();
 * #endif
 *
 * In this example, the libdarwin maintainer and API adopter agree on an API
 * version of 20180424 ahead of time for the introduction of
 * darwin_new_api_call(). When a libdarwin with that API version is submitted,
 * the project is rebuilt, and the new API becomes active.
 *
 * Breaking API changes will be both covered under this mechanism as well as
 * individual preprocessor macros in this header that declare new behavior as
 * required.
 */
#define DARWIN_API_VERSION 20210428u

#if !DARWIN_BUILDING_LIBSYSTEM_DARWIN
#define DARWIN_API_AVAILABLE_20170407 \
		API_AVAILABLE(macos(10.14), ios(12.0), tvos(12.0), watchos(5.0))
#define DARWIN_API_AVAILABLE_20180727 \
		API_AVAILABLE(macos(10.15), ios(13.0), tvos(13.0), watchos(6.0))
#define DARWIN_API_AVAILABLE_20181020 \
		API_AVAILABLE(macos(10.15), ios(13.0), tvos(13.0), watchos(6.0))
#define DARWIN_API_AVAILABLE_20181020 \
		API_AVAILABLE(macos(10.15), ios(13.0), tvos(13.0), watchos(6.0))
#define DARWIN_API_AVAILABLE_20190830 \
		API_AVAILABLE(macos(10.15.2), ios(13.3), tvos(13.3), watchos(6.1.1))
#define DARWIN_API_AVAILABLE_20191015 \
		API_AVAILABLE(macos(10.15.2), ios(13.3), tvos(13.3), watchos(6.1.1))
#define DARWIN_API_AVAILABLE_20200220 \
		API_AVAILABLE(macos(10.16), ios(14.0), tvos(14.0), watchos(7.0))
#define DARWIN_API_AVAILABLE_20200401 \
		API_AVAILABLE(macos(10.16), ios(14.0), tvos(14.0), watchos(7.0))
#define DARWIN_API_AVAILABLE_20200526 \
		API_AVAILABLE(macos(10.16), ios(14.0), tvos(14.0), watchos(7.0))
#define DARWIN_API_AVAILABLE_20210428 \
		API_AVAILABLE(macos(12.0), ios(15.0), tvos(15.0), watchos(8.0))
#else
#define DARWIN_API_AVAILABLE_20170407
#define DARWIN_API_AVAILABLE_20180727
#define DARWIN_API_AVAILABLE_20181020
#define DARWIN_API_AVAILABLE_20190830
#define DARWIN_API_AVAILABLE_20191015
#define DARWIN_API_AVAILABLE_20200220
#define DARWIN_API_AVAILABLE_20200401
#define DARWIN_API_AVAILABLE_20200526
#define DARWIN_API_AVAILABLE_20210428
#endif

/*!
 * @typedef os_struct_magic_t
 * A type representing the magic number of a transparent structure.
 */
typedef uint32_t os_struct_magic_t;

/*!
 * @typedef os_struct_version_t
 * A type representing the version of a transparent structure.
 */
typedef uint32_t os_struct_version_t;

OS_ASSUME_PTR_ABI_SINGLE_END;

#endif // __DARWIN_API_H
