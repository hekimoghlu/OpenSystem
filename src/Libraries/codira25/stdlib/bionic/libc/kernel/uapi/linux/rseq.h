/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 15, 2025.
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
#ifndef _UAPI_LINUX_RSEQ_H
#define _UAPI_LINUX_RSEQ_H
#include <linux/types.h>
#include <asm/byteorder.h>
enum rseq_cpu_id_state {
  RSEQ_CPU_ID_UNINITIALIZED = - 1,
  RSEQ_CPU_ID_REGISTRATION_FAILED = - 2,
};
enum rseq_flags {
  RSEQ_FLAG_UNREGISTER = (1 << 0),
};
enum rseq_cs_flags_bit {
  RSEQ_CS_FLAG_NO_RESTART_ON_PREEMPT_BIT = 0,
  RSEQ_CS_FLAG_NO_RESTART_ON_SIGNAL_BIT = 1,
  RSEQ_CS_FLAG_NO_RESTART_ON_MIGRATE_BIT = 2,
};
enum rseq_cs_flags {
  RSEQ_CS_FLAG_NO_RESTART_ON_PREEMPT = (1U << RSEQ_CS_FLAG_NO_RESTART_ON_PREEMPT_BIT),
  RSEQ_CS_FLAG_NO_RESTART_ON_SIGNAL = (1U << RSEQ_CS_FLAG_NO_RESTART_ON_SIGNAL_BIT),
  RSEQ_CS_FLAG_NO_RESTART_ON_MIGRATE = (1U << RSEQ_CS_FLAG_NO_RESTART_ON_MIGRATE_BIT),
};
struct rseq_cs {
  __u32 version;
  __u32 flags;
  __u64 start_ip;
  __u64 post_commit_offset;
  __u64 abort_ip;
} __attribute__((aligned(4 * sizeof(__u64))));
struct rseq {
  __u32 cpu_id_start;
  __u32 cpu_id;
  __u64 rseq_cs;
  __u32 flags;
  __u32 node_id;
  __u32 mm_cid;
  char end[];
} __attribute__((aligned(4 * sizeof(__u64))));
#endif
