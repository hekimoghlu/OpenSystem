/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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
#ifndef _UAPI_RPMSG_H_
#define _UAPI_RPMSG_H_
#include <linux/ioctl.h>
#include <linux/types.h>
#define RPMSG_ADDR_ANY 0xFFFFFFFF
struct rpmsg_endpoint_info {
  char name[32];
  __u32 src;
  __u32 dst;
};
#define RPMSG_CREATE_EPT_IOCTL _IOW(0xb5, 0x1, struct rpmsg_endpoint_info)
#define RPMSG_DESTROY_EPT_IOCTL _IO(0xb5, 0x2)
#define RPMSG_CREATE_DEV_IOCTL _IOW(0xb5, 0x3, struct rpmsg_endpoint_info)
#define RPMSG_RELEASE_DEV_IOCTL _IOW(0xb5, 0x4, struct rpmsg_endpoint_info)
#define RPMSG_GET_OUTGOING_FLOWCONTROL _IOR(0xb5, 0x5, int)
#define RPMSG_SET_INCOMING_FLOWCONTROL _IOR(0xb5, 0x6, int)
#endif
