/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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
#ifndef _UAPI_LINUX_WMI_H
#define _UAPI_LINUX_WMI_H
#include <linux/ioctl.h>
#include <linux/types.h>
#define WMI_IOC 'W'
struct wmi_ioctl_buffer {
  __u64 length;
  __u8 data[];
};
struct calling_interface_buffer {
  __u16 cmd_class;
  __u16 cmd_select;
  volatile __u32 input[4];
  volatile __u32 output[4];
} __attribute__((__packed__));
struct dell_wmi_extensions {
  __u32 argattrib;
  __u32 blength;
  __u8 data[];
} __attribute__((__packed__));
struct dell_wmi_smbios_buffer {
  __u64 length;
  struct calling_interface_buffer std;
  struct dell_wmi_extensions ext;
} __attribute__((__packed__));
#define CLASS_TOKEN_READ 0
#define CLASS_TOKEN_WRITE 1
#define SELECT_TOKEN_STD 0
#define SELECT_TOKEN_BAT 1
#define SELECT_TOKEN_AC 2
#define CLASS_FLASH_INTERFACE 7
#define SELECT_FLASH_INTERFACE 3
#define CLASS_ADMIN_PROP 10
#define SELECT_ADMIN_PROP 3
#define CLASS_INFO 17
#define SELECT_RFKILL 11
#define SELECT_APP_REGISTRATION 3
#define SELECT_DOCK 22
#define CAPSULE_EN_TOKEN 0x0461
#define CAPSULE_DIS_TOKEN 0x0462
#define WSMT_EN_TOKEN 0x04EC
#define WSMT_DIS_TOKEN 0x04ED
#define DELL_WMI_SMBIOS_CMD _IOWR(WMI_IOC, 0, struct dell_wmi_smbios_buffer)
#endif
