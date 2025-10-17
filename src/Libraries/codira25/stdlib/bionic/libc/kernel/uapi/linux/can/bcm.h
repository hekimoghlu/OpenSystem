/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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
#ifndef _UAPI_CAN_BCM_H
#define _UAPI_CAN_BCM_H
#include <linux/types.h>
#include <linux/can.h>
struct bcm_timeval {
  long tv_sec;
  long tv_usec;
};
struct bcm_msg_head {
  __u32 opcode;
  __u32 flags;
  __u32 count;
  struct bcm_timeval ival1, ival2;
  canid_t can_id;
  __u32 nframes;
  struct can_frame frames[];
};
enum {
  TX_SETUP = 1,
  TX_DELETE,
  TX_READ,
  TX_SEND,
  RX_SETUP,
  RX_DELETE,
  RX_READ,
  TX_STATUS,
  TX_EXPIRED,
  RX_STATUS,
  RX_TIMEOUT,
  RX_CHANGED
};
#define SETTIMER 0x0001
#define STARTTIMER 0x0002
#define TX_COUNTEVT 0x0004
#define TX_ANNOUNCE 0x0008
#define TX_CP_CAN_ID 0x0010
#define RX_FILTER_ID 0x0020
#define RX_CHECK_DLC 0x0040
#define RX_NO_AUTOTIMER 0x0080
#define RX_ANNOUNCE_RESUME 0x0100
#define TX_RESET_MULTI_IDX 0x0200
#define RX_RTR_FRAME 0x0400
#define CAN_FD_FRAME 0x0800
#endif
