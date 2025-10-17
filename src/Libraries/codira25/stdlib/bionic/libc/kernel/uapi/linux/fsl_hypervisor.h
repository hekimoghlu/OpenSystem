/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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
#ifndef _UAPIFSL_HYPERVISOR_H
#define _UAPIFSL_HYPERVISOR_H
#include <linux/types.h>
struct fsl_hv_ioctl_restart {
  __u32 ret;
  __u32 partition;
};
struct fsl_hv_ioctl_status {
  __u32 ret;
  __u32 partition;
  __u32 status;
};
struct fsl_hv_ioctl_start {
  __u32 ret;
  __u32 partition;
  __u32 entry_point;
  __u32 load;
};
struct fsl_hv_ioctl_stop {
  __u32 ret;
  __u32 partition;
};
struct fsl_hv_ioctl_memcpy {
  __u32 ret;
  __u32 source;
  __u32 target;
  __u32 reserved;
  __u64 local_vaddr;
  __u64 remote_paddr;
  __u64 count;
};
struct fsl_hv_ioctl_doorbell {
  __u32 ret;
  __u32 doorbell;
};
struct fsl_hv_ioctl_prop {
  __u32 ret;
  __u32 handle;
  __u64 path;
  __u64 propname;
  __u64 propval;
  __u32 proplen;
  __u32 reserved;
};
#define FSL_HV_IOCTL_TYPE 0xAF
#define FSL_HV_IOCTL_PARTITION_RESTART _IOWR(FSL_HV_IOCTL_TYPE, 1, struct fsl_hv_ioctl_restart)
#define FSL_HV_IOCTL_PARTITION_GET_STATUS _IOWR(FSL_HV_IOCTL_TYPE, 2, struct fsl_hv_ioctl_status)
#define FSL_HV_IOCTL_PARTITION_START _IOWR(FSL_HV_IOCTL_TYPE, 3, struct fsl_hv_ioctl_start)
#define FSL_HV_IOCTL_PARTITION_STOP _IOWR(FSL_HV_IOCTL_TYPE, 4, struct fsl_hv_ioctl_stop)
#define FSL_HV_IOCTL_MEMCPY _IOWR(FSL_HV_IOCTL_TYPE, 5, struct fsl_hv_ioctl_memcpy)
#define FSL_HV_IOCTL_DOORBELL _IOWR(FSL_HV_IOCTL_TYPE, 6, struct fsl_hv_ioctl_doorbell)
#define FSL_HV_IOCTL_GETPROP _IOWR(FSL_HV_IOCTL_TYPE, 7, struct fsl_hv_ioctl_prop)
#define FSL_HV_IOCTL_SETPROP _IOWR(FSL_HV_IOCTL_TYPE, 8, struct fsl_hv_ioctl_prop)
#endif
