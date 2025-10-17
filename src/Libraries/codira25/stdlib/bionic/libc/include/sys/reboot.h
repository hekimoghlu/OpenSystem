/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
 * @file sys/reboot.h
 * @brief The reboot() function.
 */

#include <sys/cdefs.h>
#include <linux/reboot.h>

__BEGIN_DECLS

/** The glibc name for the reboot() flag `LINUX_REBOOT_CMD_RESTART`. */
#define RB_AUTOBOOT LINUX_REBOOT_CMD_RESTART
/** The glibc name for the reboot() flag `LINUX_REBOOT_CMD_HALT`. */
#define RB_HALT_SYSTEM LINUX_REBOOT_CMD_HALT
/** The glibc name for the reboot() flag `LINUX_REBOOT_CMD_CAD_ON`. */
#define RB_ENABLE_CAD LINUX_REBOOT_CMD_CAD_ON
/** The glibc name for the reboot() flag `LINUX_REBOOT_CMD_CAD_OFF`. */
#define RB_DISABLE_CAD LINUX_REBOOT_CMD_CAD_OFF
/** The glibc name for the reboot() flag `LINUX_REBOOT_CMD_POWER_OFF`. */
#define RB_POWER_OFF LINUX_REBOOT_CMD_POWER_OFF

/**
 * [reboot(2)](https://man7.org/linux/man-pages/man2/reboot.2.html) reboots the device.
 *
 * Does not return on successful reboot, returns 0 if CAD was successfully enabled/disabled,
 * and returns -1 and sets `errno` on failure.
 */
int reboot(int __op);

__END_DECLS
