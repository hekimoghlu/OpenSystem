/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 8, 2024.
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
#ifndef __LINUX__AIO_ABI_H
#define __LINUX__AIO_ABI_H
#include <linux/types.h>
#include <linux/fs.h>
#include <asm/byteorder.h>
typedef __kernel_ulong_t aio_context_t;
enum {
  IOCB_CMD_PREAD = 0,
  IOCB_CMD_PWRITE = 1,
  IOCB_CMD_FSYNC = 2,
  IOCB_CMD_FDSYNC = 3,
  IOCB_CMD_POLL = 5,
  IOCB_CMD_NOOP = 6,
  IOCB_CMD_PREADV = 7,
  IOCB_CMD_PWRITEV = 8,
};
#define IOCB_FLAG_RESFD (1 << 0)
#define IOCB_FLAG_IOPRIO (1 << 1)
struct io_event {
  __u64 data;
  __u64 obj;
  __s64 res;
  __s64 res2;
};
struct iocb {
  __u64 aio_data;
#if defined(__BYTE_ORDER) ? __BYTE_ORDER == __LITTLE_ENDIAN : defined(__LITTLE_ENDIAN)
  __u32 aio_key;
  __kernel_rwf_t aio_rw_flags;
#elif defined(__BYTE_ORDER)?__BYTE_ORDER==__BIG_ENDIAN:defined(__BIG_ENDIAN)
  __kernel_rwf_t aio_rw_flags;
  __u32 aio_key;
#else
#error edit for your odd byteorder .
#endif
  __u16 aio_lio_opcode;
  __s16 aio_reqprio;
  __u32 aio_fildes;
  __u64 aio_buf;
  __u64 aio_nbytes;
  __s64 aio_offset;
  __u64 aio_reserved2;
  __u32 aio_flags;
  __u32 aio_resfd;
};
#undef IFBIG
#undef IFLITTLE
#endif
