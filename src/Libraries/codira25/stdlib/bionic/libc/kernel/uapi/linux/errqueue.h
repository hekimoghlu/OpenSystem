/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
#ifndef _UAPI_LINUX_ERRQUEUE_H
#define _UAPI_LINUX_ERRQUEUE_H
#include <linux/types.h>
#include <linux/time_types.h>
struct sock_ee_data_rfc4884 {
  __u16 len;
  __u8 flags;
  __u8 reserved;
};
struct sock_extended_err {
  __u32 ee_errno;
  __u8 ee_origin;
  __u8 ee_type;
  __u8 ee_code;
  __u8 ee_pad;
  __u32 ee_info;
  union {
    __u32 ee_data;
    struct sock_ee_data_rfc4884 ee_rfc4884;
  };
};
#define SO_EE_ORIGIN_NONE 0
#define SO_EE_ORIGIN_LOCAL 1
#define SO_EE_ORIGIN_ICMP 2
#define SO_EE_ORIGIN_ICMP6 3
#define SO_EE_ORIGIN_TXSTATUS 4
#define SO_EE_ORIGIN_ZEROCOPY 5
#define SO_EE_ORIGIN_TXTIME 6
#define SO_EE_ORIGIN_TIMESTAMPING SO_EE_ORIGIN_TXSTATUS
#define SO_EE_OFFENDER(ee) ((struct sockaddr *) ((ee) + 1))
#define SO_EE_CODE_ZEROCOPY_COPIED 1
#define SO_EE_CODE_TXTIME_INVALID_PARAM 1
#define SO_EE_CODE_TXTIME_MISSED 2
#define SO_EE_RFC4884_FLAG_INVALID 1
struct scm_timestamping {
  struct timespec ts[3];
};
struct scm_timestamping64 {
  struct __kernel_timespec ts[3];
};
enum {
  SCM_TSTAMP_SND,
  SCM_TSTAMP_SCHED,
  SCM_TSTAMP_ACK,
};
#endif
