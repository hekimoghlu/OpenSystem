/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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
#ifndef _UAPI_IIO_BUFFER_H_
#define _UAPI_IIO_BUFFER_H_
#include <linux/types.h>
#define IIO_BUFFER_DMABUF_CYCLIC (1 << 0)
#define IIO_BUFFER_DMABUF_SUPPORTED_FLAGS 0x00000001
struct iio_dmabuf {
  __u32 fd;
  __u32 flags;
  __u64 bytes_used;
};
#define IIO_BUFFER_GET_FD_IOCTL _IOWR('i', 0x91, int)
#define IIO_BUFFER_DMABUF_ATTACH_IOCTL _IOW('i', 0x92, int)
#define IIO_BUFFER_DMABUF_DETACH_IOCTL _IOW('i', 0x93, int)
#define IIO_BUFFER_DMABUF_ENQUEUE_IOCTL _IOW('i', 0x94, struct iio_dmabuf)
#endif
