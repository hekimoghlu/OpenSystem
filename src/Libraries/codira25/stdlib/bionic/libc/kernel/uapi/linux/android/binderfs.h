/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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
#ifndef _UAPI_LINUX_BINDERFS_H
#define _UAPI_LINUX_BINDERFS_H
#include <linux/android/binder.h>
#include <linux/types.h>
#include <linux/ioctl.h>
#define BINDERFS_MAX_NAME 255
struct binderfs_device {
  char name[BINDERFS_MAX_NAME + 1];
  __u32 major;
  __u32 minor;
};
#define BINDER_CTL_ADD _IOWR('b', 1, struct binderfs_device)
#endif
