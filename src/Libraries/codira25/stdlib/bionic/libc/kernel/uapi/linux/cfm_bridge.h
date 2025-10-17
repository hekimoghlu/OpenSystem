/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
#ifndef _UAPI_LINUX_CFM_BRIDGE_H_
#define _UAPI_LINUX_CFM_BRIDGE_H_
#include <linux/types.h>
#include <linux/if_ether.h>
#define ETHER_HEADER_LENGTH (6 + 6 + 4 + 2)
#define CFM_MAID_LENGTH 48
#define CFM_CCM_PDU_LENGTH 75
#define CFM_PORT_STATUS_TLV_LENGTH 4
#define CFM_IF_STATUS_TLV_LENGTH 4
#define CFM_IF_STATUS_TLV_TYPE 4
#define CFM_PORT_STATUS_TLV_TYPE 2
#define CFM_ENDE_TLV_TYPE 0
#define CFM_CCM_MAX_FRAME_LENGTH (ETHER_HEADER_LENGTH + CFM_CCM_PDU_LENGTH + CFM_PORT_STATUS_TLV_LENGTH + CFM_IF_STATUS_TLV_LENGTH)
#define CFM_FRAME_PRIO 7
#define CFM_CCM_TLV_OFFSET 70
#define CFM_CCM_PDU_MAID_OFFSET 10
#define CFM_CCM_PDU_MEPID_OFFSET 8
#define CFM_CCM_PDU_SEQNR_OFFSET 4
#define CFM_CCM_PDU_TLV_OFFSET 74
#define CFM_CCM_ITU_RESERVED_SIZE 16
struct br_cfm_common_hdr {
  __u8 mdlevel_version;
  __u8 opcode;
  __u8 flags;
  __u8 tlv_offset;
};
enum br_cfm_opcodes {
  BR_CFM_OPCODE_CCM = 0x1,
};
enum br_cfm_domain {
  BR_CFM_PORT,
  BR_CFM_VLAN,
};
enum br_cfm_mep_direction {
  BR_CFM_MEP_DIRECTION_DOWN,
  BR_CFM_MEP_DIRECTION_UP,
};
enum br_cfm_ccm_interval {
  BR_CFM_CCM_INTERVAL_NONE,
  BR_CFM_CCM_INTERVAL_3_3_MS,
  BR_CFM_CCM_INTERVAL_10_MS,
  BR_CFM_CCM_INTERVAL_100_MS,
  BR_CFM_CCM_INTERVAL_1_SEC,
  BR_CFM_CCM_INTERVAL_10_SEC,
  BR_CFM_CCM_INTERVAL_1_MIN,
  BR_CFM_CCM_INTERVAL_10_MIN,
};
#endif
