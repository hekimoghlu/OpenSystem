/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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
#ifndef _SYS_MEMORY_MAINTENANCE_H_
#define _SYS_MEMORY_MAINTENANCE_H_

/*
 * File:	sys/memory_maintenance.h
 * Author:	Samuel Gosselin [sgosselin@apple.com]
 *
 * Header file for Memory Maintenance support.
 */

/*
 * The kern.darkboot sysctl can be controlled from kexts or userspace. If
 * processes want to change the sysctl value, they require the
 * 'com.apple.private.kernel.darkboot' entitlement.
 *
 * Operating the kern.darkboot sysctl is done via using the commands below:
 *
 *      - MEMORY_MAINTENANCE_DARK_BOOT_UNSET
 *              Unset the kern.darkboot sysctl (kern.sysctl=0).
 *      - MEMORY_MAINTENANCE_DARK_BOOT_SET
 *              Set the kern.darkboot sysctl (kern.sysctl=1).
 *      - MEMORY_MAINTENANCE_DARK_BOOT_SET_PERSISTENT
 *              Set the kern.darkboot sysctl (kern.sysctl=1) and save its
 *              value into the 'darkboot' NVRAM variable.
 *
 * Example:
 *      sysctl kern.darkboot=2
 */
#define MEMORY_MAINTENANCE_DARK_BOOT_UNSET              (0)
#define MEMORY_MAINTENANCE_DARK_BOOT_SET                (1)
#define MEMORY_MAINTENANCE_DARK_BOOT_SET_PERSISTENT     (2)

#define MEMORY_MAINTENANCE_DARK_BOOT_NVRAM_NAME         "darkboot"

#endif /* _SYS_MEMORY_MAINTENANCE_H_ */
