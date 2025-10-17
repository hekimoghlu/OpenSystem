/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#ifndef _VFIO_ZDEV_H_
#define _VFIO_ZDEV_H_
#include <linux/types.h>
#include <linux/vfio.h>
struct vfio_device_info_cap_zpci_base {
  struct vfio_info_cap_header header;
  __u64 start_dma;
  __u64 end_dma;
  __u16 pchid;
  __u16 vfn;
  __u16 fmb_length;
  __u8 pft;
  __u8 gid;
  __u32 fh;
};
struct vfio_device_info_cap_zpci_group {
  struct vfio_info_cap_header header;
  __u64 dasm;
  __u64 msi_addr;
  __u64 flags;
#define VFIO_DEVICE_INFO_ZPCI_FLAG_REFRESH 1
  __u16 mui;
  __u16 noi;
  __u16 maxstbl;
  __u8 version;
  __u8 reserved;
  __u16 imaxstbl;
};
struct vfio_device_info_cap_zpci_util {
  struct vfio_info_cap_header header;
  __u32 size;
  __u8 util_str[];
};
struct vfio_device_info_cap_zpci_pfip {
  struct vfio_info_cap_header header;
  __u32 size;
  __u8 pfip[];
};
#endif
