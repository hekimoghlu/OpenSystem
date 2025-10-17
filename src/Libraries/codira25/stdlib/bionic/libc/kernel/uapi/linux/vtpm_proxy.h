/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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
#ifndef _UAPI_LINUX_VTPM_PROXY_H
#define _UAPI_LINUX_VTPM_PROXY_H
#include <linux/types.h>
#include <linux/ioctl.h>
enum vtpm_proxy_flags {
  VTPM_PROXY_FLAG_TPM2 = 1,
};
struct vtpm_proxy_new_dev {
  __u32 flags;
  __u32 tpm_num;
  __u32 fd;
  __u32 major;
  __u32 minor;
};
#define VTPM_PROXY_IOC_NEW_DEV _IOWR(0xa1, 0x00, struct vtpm_proxy_new_dev)
#define TPM2_CC_SET_LOCALITY 0x20001000
#define TPM_ORD_SET_LOCALITY 0x20001000
#endif
