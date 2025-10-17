/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 7, 2021.
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
#ifndef _UAPI_LINUX_IF_VLAN_H_
#define _UAPI_LINUX_IF_VLAN_H_
enum vlan_ioctl_cmds {
  ADD_VLAN_CMD,
  DEL_VLAN_CMD,
  SET_VLAN_INGRESS_PRIORITY_CMD,
  SET_VLAN_EGRESS_PRIORITY_CMD,
  GET_VLAN_INGRESS_PRIORITY_CMD,
  GET_VLAN_EGRESS_PRIORITY_CMD,
  SET_VLAN_NAME_TYPE_CMD,
  SET_VLAN_FLAG_CMD,
  GET_VLAN_REALDEV_NAME_CMD,
  GET_VLAN_VID_CMD
};
enum vlan_flags {
  VLAN_FLAG_REORDER_HDR = 0x1,
  VLAN_FLAG_GVRP = 0x2,
  VLAN_FLAG_LOOSE_BINDING = 0x4,
  VLAN_FLAG_MVRP = 0x8,
  VLAN_FLAG_BRIDGE_BINDING = 0x10,
};
enum vlan_name_types {
  VLAN_NAME_TYPE_PLUS_VID,
  VLAN_NAME_TYPE_RAW_PLUS_VID,
  VLAN_NAME_TYPE_PLUS_VID_NO_PAD,
  VLAN_NAME_TYPE_RAW_PLUS_VID_NO_PAD,
  VLAN_NAME_TYPE_HIGHEST
};
struct vlan_ioctl_args {
  int cmd;
  char device1[24];
  union {
    char device2[24];
    int VID;
    unsigned int skb_priority;
    unsigned int name_type;
    unsigned int bind_type;
    unsigned int flag;
  } u;
  short vlan_qos;
};
#endif
