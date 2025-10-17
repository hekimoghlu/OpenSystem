/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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
#ifndef __REMOTE_VIF_H__
#define __REMOTE_VIF_H__

#include <sys/proc.h>
#include <net/if.h>
#include <net/bpf.h>

#include <net/pktap.h>

#define RVI_CONTROL_NAME        "com.apple.net.rvi_control"
#define RVI_BUFFERSZ            (64 * 1024)
#define RVI_VERSION_1           0x1
#define RVI_VERSION_2           0x2
#define RVI_VERSION_CURRENT     RVI_VERSION_2

enum  {
	RVI_COMMAND_OUT_PAYLOAD         = 0x01,
	RVI_COMMAND_IN_PAYLOAD          = 0x10,
	RVI_COMMAND_GET_INTERFACE       = 0x20,
	RVI_COMMAND_VERSION             = 0x40
};

#ifdef XNU_KERNEL_PRIVATE
int rvi_init(void);
#endif /* XNU_KERNEL_PRIVATE */

#endif /* __REMOTE_VIF_H__ */
