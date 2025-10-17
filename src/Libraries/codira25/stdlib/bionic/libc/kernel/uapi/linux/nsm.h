/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 20, 2022.
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
#ifndef __UAPI_LINUX_NSM_H
#define __UAPI_LINUX_NSM_H
#include <linux/ioctl.h>
#include <linux/types.h>
#define NSM_MAGIC 0x0A
#define NSM_REQUEST_MAX_SIZE 0x1000
#define NSM_RESPONSE_MAX_SIZE 0x3000
struct nsm_iovec {
  __u64 addr;
  __u64 len;
};
struct nsm_raw {
  struct nsm_iovec request;
  struct nsm_iovec response;
};
#define NSM_IOCTL_RAW _IOWR(NSM_MAGIC, 0x0, struct nsm_raw)
#endif
