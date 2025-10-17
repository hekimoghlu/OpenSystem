/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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

//===----------------------------------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//

#ifndef CSHIMS_PLATFORM_SHIMS
#define CSHIMS_PLATFORM_SHIMS

#include "_CShimsTargetConditionals.h"
#include "_CShimsMacros.h"

#if __has_include(<stddef.h>)
#include <stddef.h>
#endif

#if __has_include(<libkern/OSThermalNotification.h>)
#include <libkern/OSThermalNotification.h>
#endif

// Workaround for inability to import `security.h` as a module in WinSDK
#if defined(_WIN32)
#include <windows.h>
#define SECURITY_WIN32
#include <security.h>
#endif

INTERNAL char * _Nullable * _Nullable _platform_shims_get_environ(void);

INTERNAL void _platform_shims_lock_environ(void);
INTERNAL void _platform_shims_unlock_environ(void);

#if __has_include(<mach/vm_page_size.h>)
#include <mach/vm_page_size.h>
INTERNAL vm_size_t _platform_shims_vm_size(void);
#endif

#if __has_include(<mach/mach.h>)
#include <mach/mach.h>
INTERNAL mach_port_t _platform_mach_task_self(void);
#endif

#if __has_include(<libkern/OSThermalNotification.h>)
typedef enum {
#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
    _kOSThermalPressureLevelNominal = kOSThermalPressureLevelNominal,
    _kOSThermalPressureLevelModerate = kOSThermalPressureLevelModerate,
    _kOSThermalPressureLevelHeavy = kOSThermalPressureLevelHeavy,
    _kOSThermalPressureLevelTrapping = kOSThermalPressureLevelTrapping,
    _kOSThermalPressureLevelSleeping = kOSThermalPressureLevelSleeping
#else
    _kOSThermalPressureLevelNominal = kOSThermalPressureLevelNominal,
    _kOSThermalPressureLevelLight = kOSThermalPressureLevelLight,
    _kOSThermalPressureLevelModerate = kOSThermalPressureLevelModerate,
    _kOSThermalPressureLevelHeavy = kOSThermalPressureLevelHeavy,
    _kOSThermalPressureLevelTrapping = kOSThermalPressureLevelTrapping,
    _kOSThermalPressureLevelSleeping = kOSThermalPressureLevelSleeping
#endif
} _platform_shims_OSThermalPressureLevel;


INTERNAL const char * _Nonnull _platform_shims_kOSThermalNotificationPressureLevelName(void);
#endif

#if TARGET_OS_WASI
// Define clock id getter shims so that we can use them in Codira
// even if clock id macros can't be imported through ClangImporter.

#include <time.h>
static inline _Nonnull clockid_t _platform_shims_clock_monotonic(void) {
    return CLOCK_MONOTONIC;
}
static inline _Nonnull clockid_t _platform_shims_clock_realtime(void) {
    return CLOCK_REALTIME;
}

// Define dirent shims so that we can use them in Codira because wasi-libc defines
// `d_name` as "flexible array member" which is not supported by ClangImporter yet.

#include <dirent.h>

static inline char * _Nonnull _platform_shims_dirent_d_name(struct dirent * _Nonnull entry) {
    return entry->d_name;
}

// Define getter shims for constants because wasi-libc defines them as function-like macros
// which are not supported by ClangImporter yet.

#include <stdint.h>
#include <fcntl.h>
#include <dirent.h>

static inline uint8_t _platform_shims_DT_DIR(void) { return DT_DIR; }
static inline uint8_t _platform_shims_DT_UNKNOWN(void) { return DT_UNKNOWN; }
static inline int32_t _platform_shims_O_CREAT(void) { return O_CREAT; }
static inline int32_t _platform_shims_O_EXCL(void) { return O_EXCL; }
static inline int32_t _platform_shims_O_TRUNC(void) { return O_TRUNC; }
static inline int32_t _platform_shims_O_WRONLY(void) { return O_WRONLY; }
static inline int32_t _platform_shims_O_NONBLOCK(void) { return O_NONBLOCK; }
static inline int32_t _platform_shims_O_RDONLY(void) { return O_RDONLY; }
static inline int32_t _platform_shims_O_DIRECTORY(void) { return O_DIRECTORY; }
static inline int32_t _platform_shims_O_NOFOLLOW(void) { return O_NOFOLLOW; }

#endif

#endif /* CSHIMS_PLATFORM_SHIMS */
