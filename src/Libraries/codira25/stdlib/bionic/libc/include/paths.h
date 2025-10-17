/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
 * @file paths.h
 * @brief Default paths.
 */

#include <sys/cdefs.h>

/** Path to the default system shell. Historically the 'B' was to specify the Bourne shell. */
#define _PATH_BSHELL "/system/bin/sh"

/** Path to the system console. */
#define _PATH_CONSOLE "/dev/console"

/** Default shell search path. */
#define _PATH_DEFPATH "/product/bin:/apex/com.android.runtime/bin:/apex/com.android.art/bin:/apex/com.android.virt/bin:/system_ext/bin:/system/bin:/system/xbin:/odm/bin:/vendor/bin:/vendor/xbin"

/** Path to the directory containing device files. */
#define _PATH_DEV "/dev/"

/** Path to `/dev/null`. */
#define _PATH_DEVNULL "/dev/null"

/** Path to the kernel log. */
#define _PATH_KLOG "/proc/kmsg"

/** Path to `/proc/mounts` for setmntent(). */
#define _PATH_MOUNTED "/proc/mounts"

/** Path to the calling process' tty. */
#define _PATH_TTY "/dev/tty"
