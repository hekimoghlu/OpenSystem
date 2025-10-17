/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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
#ifndef _UAPI_LINUX_ACCT_H
#define _UAPI_LINUX_ACCT_H
#include <linux/types.h>
#include <asm/param.h>
#include <asm/byteorder.h>
typedef __u16 comp_t;
typedef __u32 comp2_t;
#define ACCT_COMM 16
struct acct {
  char ac_flag;
  char ac_version;
  __u16 ac_uid16;
  __u16 ac_gid16;
  __u16 ac_tty;
  __u32 ac_btime;
  comp_t ac_utime;
  comp_t ac_stime;
  comp_t ac_etime;
  comp_t ac_mem;
  comp_t ac_io;
  comp_t ac_rw;
  comp_t ac_minflt;
  comp_t ac_majflt;
  comp_t ac_swaps;
  __u16 ac_ahz;
  __u32 ac_exitcode;
  char ac_comm[ACCT_COMM + 1];
  __u8 ac_etime_hi;
  __u16 ac_etime_lo;
  __u32 ac_uid;
  __u32 ac_gid;
};
struct acct_v3 {
  char ac_flag;
  char ac_version;
  __u16 ac_tty;
  __u32 ac_exitcode;
  __u32 ac_uid;
  __u32 ac_gid;
  __u32 ac_pid;
  __u32 ac_ppid;
  __u32 ac_btime;
  float ac_etime;
  comp_t ac_utime;
  comp_t ac_stime;
  comp_t ac_mem;
  comp_t ac_io;
  comp_t ac_rw;
  comp_t ac_minflt;
  comp_t ac_majflt;
  comp_t ac_swaps;
  char ac_comm[ACCT_COMM];
};
#define AFORK 0x01
#define ASU 0x02
#define ACOMPAT 0x04
#define ACORE 0x08
#define AXSIG 0x10
#define AGROUP 0x20
#if defined(__BYTE_ORDER) ? __BYTE_ORDER == __BIG_ENDIAN : defined(__BIG_ENDIAN)
#define ACCT_BYTEORDER 0x80
#elif defined(__BYTE_ORDER)?__BYTE_ORDER==__LITTLE_ENDIAN:defined(__LITTLE_ENDIAN)
#define ACCT_BYTEORDER 0x00
#else
#error unspecified endianness
#endif
#define ACCT_VERSION 2
#define AHZ (HZ)
#endif
