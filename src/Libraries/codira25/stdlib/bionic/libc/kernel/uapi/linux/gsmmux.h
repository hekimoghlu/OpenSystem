/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#ifndef _LINUX_GSMMUX_H
#define _LINUX_GSMMUX_H
#include <linux/const.h>
#include <linux/if.h>
#include <linux/ioctl.h>
#include <linux/types.h>
#define GSM_FL_RESTART _BITUL(0)
struct gsm_config {
  unsigned int adaption;
  unsigned int encapsulation;
  unsigned int initiator;
  unsigned int t1;
  unsigned int t2;
  unsigned int t3;
  unsigned int n2;
  unsigned int mru;
  unsigned int mtu;
  unsigned int k;
  unsigned int i;
  unsigned int unused[8];
};
#define GSMIOC_GETCONF _IOR('G', 0, struct gsm_config)
#define GSMIOC_SETCONF _IOW('G', 1, struct gsm_config)
struct gsm_netconfig {
  unsigned int adaption;
  unsigned short protocol;
  unsigned short unused2;
  char if_name[IFNAMSIZ];
  __u8 unused[28];
};
#define GSMIOC_ENABLE_NET _IOW('G', 2, struct gsm_netconfig)
#define GSMIOC_DISABLE_NET _IO('G', 3)
#define GSMIOC_GETFIRST _IOR('G', 4, __u32)
struct gsm_config_ext {
  __u32 keep_alive;
  __u32 wait_config;
  __u32 flags;
  __u32 reserved[5];
};
#define GSMIOC_GETCONF_EXT _IOR('G', 5, struct gsm_config_ext)
#define GSMIOC_SETCONF_EXT _IOW('G', 6, struct gsm_config_ext)
struct gsm_dlci_config {
  __u32 channel;
  __u32 adaption;
  __u32 mtu;
  __u32 priority;
  __u32 i;
  __u32 k;
  __u32 flags;
  __u32 reserved[7];
};
#define GSMIOC_GETCONF_DLCI _IOWR('G', 7, struct gsm_dlci_config)
#define GSMIOC_SETCONF_DLCI _IOW('G', 8, struct gsm_dlci_config)
#endif
