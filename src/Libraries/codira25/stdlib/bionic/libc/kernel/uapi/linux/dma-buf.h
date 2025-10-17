/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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
#ifndef _DMA_BUF_UAPI_H_
#define _DMA_BUF_UAPI_H_
#include <linux/types.h>
struct dma_buf_sync {
  __u64 flags;
};
#define DMA_BUF_SYNC_READ (1 << 0)
#define DMA_BUF_SYNC_WRITE (2 << 0)
#define DMA_BUF_SYNC_RW (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_SYNC_START (0 << 2)
#define DMA_BUF_SYNC_END (1 << 2)
#define DMA_BUF_SYNC_VALID_FLAGS_MASK (DMA_BUF_SYNC_RW | DMA_BUF_SYNC_END)
#define DMA_BUF_NAME_LEN 32
struct dma_buf_export_sync_file {
  __u32 flags;
  __s32 fd;
};
struct dma_buf_import_sync_file {
  __u32 flags;
  __s32 fd;
};
#define DMA_BUF_BASE 'b'
#define DMA_BUF_IOCTL_SYNC _IOW(DMA_BUF_BASE, 0, struct dma_buf_sync)
#define DMA_BUF_SET_NAME _IOW(DMA_BUF_BASE, 1, const char *)
#define DMA_BUF_SET_NAME_A _IOW(DMA_BUF_BASE, 1, __u32)
#define DMA_BUF_SET_NAME_B _IOW(DMA_BUF_BASE, 1, __u64)
#define DMA_BUF_IOCTL_EXPORT_SYNC_FILE _IOWR(DMA_BUF_BASE, 2, struct dma_buf_export_sync_file)
#define DMA_BUF_IOCTL_IMPORT_SYNC_FILE _IOW(DMA_BUF_BASE, 3, struct dma_buf_import_sync_file)
#endif
