/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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
#ifndef _UAPI_ASM_X86_MCE_H
#define _UAPI_ASM_X86_MCE_H
#include <linux/types.h>
#include <linux/ioctl.h>
struct mce {
  __u64 status;
  __u64 misc;
  __u64 addr;
  __u64 mcgstatus;
  __u64 ip;
  __u64 tsc;
  __u64 time;
  __u8 cpuvendor;
  __u8 inject_flags;
  __u8 severity;
  __u8 pad;
  __u32 cpuid;
  __u8 cs;
  __u8 bank;
  __u8 cpu;
  __u8 finished;
  __u32 extcpu;
  __u32 socketid;
  __u32 apicid;
  __u64 mcgcap;
  __u64 synd;
  __u64 ipid;
  __u64 ppin;
  __u32 microcode;
  __u64 kflags;
};
#define MCE_GET_RECORD_LEN _IOR('M', 1, int)
#define MCE_GET_LOG_LEN _IOR('M', 2, int)
#define MCE_GETCLEAR_FLAGS _IOR('M', 3, int)
#endif
