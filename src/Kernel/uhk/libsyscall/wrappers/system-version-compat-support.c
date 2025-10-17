/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
#include "system-version-compat-support.h"

#if SYSTEM_VERSION_COMPAT_ENABLED


#include <stdbool.h>
#include <sys/param.h>
#include <sys/types.h>

__attribute__((visibility("hidden")))
bool (*system_version_compat_check_path_suffix)(const char *orig_path) = NULL;
system_version_compat_mode_t system_version_compat_mode = SYSTEM_VERSION_COMPAT_MODE_DISABLED;

__attribute__((visibility("hidden")))
int (*system_version_compat_open_shim)(int opened_fd, int openat_fd, const char *orig_path, int oflag, mode_t mode,
    int (*close_syscall)(int), int (*open_syscall)(const char *, int, mode_t),
    int (*openat_syscall)(int, const char *, int, mode_t),
    int (*fcntl_syscall)(int, int, long)) = NULL;
#endif /* SYSTEM_VERSION_COMPAT_ENABLED */
