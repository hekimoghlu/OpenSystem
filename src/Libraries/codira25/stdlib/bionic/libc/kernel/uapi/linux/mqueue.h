/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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
#ifndef _LINUX_MQUEUE_H
#define _LINUX_MQUEUE_H
#include <linux/types.h>
#define MQ_PRIO_MAX 32768
#define MQ_BYTES_MAX 819200
struct mq_attr {
  __kernel_long_t mq_flags;
  __kernel_long_t mq_maxmsg;
  __kernel_long_t mq_msgsize;
  __kernel_long_t mq_curmsgs;
  __kernel_long_t __reserved[4];
};
#define NOTIFY_NONE 0
#define NOTIFY_WOKENUP 1
#define NOTIFY_REMOVED 2
#define NOTIFY_COOKIE_LEN 32
#endif
