/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
#ifndef _UAPI_REMOTEPROC_CDEV_H_
#define _UAPI_REMOTEPROC_CDEV_H_
#include <linux/ioctl.h>
#include <linux/types.h>
#define RPROC_MAGIC 0xB7
#define RPROC_SET_SHUTDOWN_ON_RELEASE _IOW(RPROC_MAGIC, 1, __s32)
#define RPROC_GET_SHUTDOWN_ON_RELEASE _IOR(RPROC_MAGIC, 2, __s32)
#endif
