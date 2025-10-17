/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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
#ifndef _UAPI_CAN_RAW_H
#define _UAPI_CAN_RAW_H
#include <linux/can.h>
#define SOL_CAN_RAW (SOL_CAN_BASE + CAN_RAW)
#define CAN_RAW_FILTER_MAX 512
enum {
  SCM_CAN_RAW_ERRQUEUE = 1,
};
enum {
  CAN_RAW_FILTER = 1,
  CAN_RAW_ERR_FILTER,
  CAN_RAW_LOOPBACK,
  CAN_RAW_RECV_OWN_MSGS,
  CAN_RAW_FD_FRAMES,
  CAN_RAW_JOIN_FILTERS,
  CAN_RAW_XL_FRAMES,
  CAN_RAW_XL_VCID_OPTS,
};
struct can_raw_vcid_options {
  __u8 flags;
  __u8 tx_vcid;
  __u8 rx_vcid;
  __u8 rx_vcid_mask;
};
#define CAN_RAW_XL_VCID_TX_SET 0x01
#define CAN_RAW_XL_VCID_TX_PASS 0x02
#define CAN_RAW_XL_VCID_RX_FILTER 0x04
#endif
