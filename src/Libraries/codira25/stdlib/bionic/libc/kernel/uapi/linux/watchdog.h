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
#ifndef _UAPI_LINUX_WATCHDOG_H
#define _UAPI_LINUX_WATCHDOG_H
#include <linux/ioctl.h>
#include <linux/types.h>
#define WATCHDOG_IOCTL_BASE 'W'
struct watchdog_info {
  __u32 options;
  __u32 firmware_version;
  __u8 identity[32];
};
#define WDIOC_GETSUPPORT _IOR(WATCHDOG_IOCTL_BASE, 0, struct watchdog_info)
#define WDIOC_GETSTATUS _IOR(WATCHDOG_IOCTL_BASE, 1, int)
#define WDIOC_GETBOOTSTATUS _IOR(WATCHDOG_IOCTL_BASE, 2, int)
#define WDIOC_GETTEMP _IOR(WATCHDOG_IOCTL_BASE, 3, int)
#define WDIOC_SETOPTIONS _IOR(WATCHDOG_IOCTL_BASE, 4, int)
#define WDIOC_KEEPALIVE _IOR(WATCHDOG_IOCTL_BASE, 5, int)
#define WDIOC_SETTIMEOUT _IOWR(WATCHDOG_IOCTL_BASE, 6, int)
#define WDIOC_GETTIMEOUT _IOR(WATCHDOG_IOCTL_BASE, 7, int)
#define WDIOC_SETPRETIMEOUT _IOWR(WATCHDOG_IOCTL_BASE, 8, int)
#define WDIOC_GETPRETIMEOUT _IOR(WATCHDOG_IOCTL_BASE, 9, int)
#define WDIOC_GETTIMELEFT _IOR(WATCHDOG_IOCTL_BASE, 10, int)
#define WDIOF_UNKNOWN - 1
#define WDIOS_UNKNOWN - 1
#define WDIOF_OVERHEAT 0x0001
#define WDIOF_FANFAULT 0x0002
#define WDIOF_EXTERN1 0x0004
#define WDIOF_EXTERN2 0x0008
#define WDIOF_POWERUNDER 0x0010
#define WDIOF_CARDRESET 0x0020
#define WDIOF_POWEROVER 0x0040
#define WDIOF_SETTIMEOUT 0x0080
#define WDIOF_MAGICCLOSE 0x0100
#define WDIOF_PRETIMEOUT 0x0200
#define WDIOF_ALARMONLY 0x0400
#define WDIOF_KEEPALIVEPING 0x8000
#define WDIOS_DISABLECARD 0x0001
#define WDIOS_ENABLECARD 0x0002
#define WDIOS_TEMPPANIC 0x0004
#endif
