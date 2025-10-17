/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
#ifndef _UAPI_LINUX_STM_H
#define _UAPI_LINUX_STM_H
#include <linux/types.h>
#define STP_MASTER_MAX 0xffff
#define STP_CHANNEL_MAX 0xffff
struct stp_policy_id {
  __u32 size;
  __u16 master;
  __u16 channel;
  __u16 width;
  __u16 __reserved_0;
  __u32 __reserved_1;
  char id[];
};
#define STP_POLICY_ID_SET _IOWR('%', 0, struct stp_policy_id)
#define STP_POLICY_ID_GET _IOR('%', 1, struct stp_policy_id)
#define STP_SET_OPTIONS _IOW('%', 2, __u64)
#endif
