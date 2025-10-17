/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#ifndef _UAPI__LINUX_GENERIC_NETLINK_H
#define _UAPI__LINUX_GENERIC_NETLINK_H
#include <linux/types.h>
#include <linux/netlink.h>
#define GENL_NAMSIZ 16
#define GENL_MIN_ID NLMSG_MIN_TYPE
#define GENL_MAX_ID 1023
struct genlmsghdr {
  __u8 cmd;
  __u8 version;
  __u16 reserved;
};
#define GENL_HDRLEN NLMSG_ALIGN(sizeof(struct genlmsghdr))
#define GENL_ADMIN_PERM 0x01
#define GENL_CMD_CAP_DO 0x02
#define GENL_CMD_CAP_DUMP 0x04
#define GENL_CMD_CAP_HASPOL 0x08
#define GENL_UNS_ADMIN_PERM 0x10
#define GENL_ID_CTRL NLMSG_MIN_TYPE
#define GENL_ID_VFS_DQUOT (NLMSG_MIN_TYPE + 1)
#define GENL_ID_PMCRAID (NLMSG_MIN_TYPE + 2)
#define GENL_START_ALLOC (NLMSG_MIN_TYPE + 3)
enum {
  CTRL_CMD_UNSPEC,
  CTRL_CMD_NEWFAMILY,
  CTRL_CMD_DELFAMILY,
  CTRL_CMD_GETFAMILY,
  CTRL_CMD_NEWOPS,
  CTRL_CMD_DELOPS,
  CTRL_CMD_GETOPS,
  CTRL_CMD_NEWMCAST_GRP,
  CTRL_CMD_DELMCAST_GRP,
  CTRL_CMD_GETMCAST_GRP,
  CTRL_CMD_GETPOLICY,
  __CTRL_CMD_MAX,
};
#define CTRL_CMD_MAX (__CTRL_CMD_MAX - 1)
enum {
  CTRL_ATTR_UNSPEC,
  CTRL_ATTR_FAMILY_ID,
  CTRL_ATTR_FAMILY_NAME,
  CTRL_ATTR_VERSION,
  CTRL_ATTR_HDRSIZE,
  CTRL_ATTR_MAXATTR,
  CTRL_ATTR_OPS,
  CTRL_ATTR_MCAST_GROUPS,
  CTRL_ATTR_POLICY,
  CTRL_ATTR_OP_POLICY,
  CTRL_ATTR_OP,
  __CTRL_ATTR_MAX,
};
#define CTRL_ATTR_MAX (__CTRL_ATTR_MAX - 1)
enum {
  CTRL_ATTR_OP_UNSPEC,
  CTRL_ATTR_OP_ID,
  CTRL_ATTR_OP_FLAGS,
  __CTRL_ATTR_OP_MAX,
};
#define CTRL_ATTR_OP_MAX (__CTRL_ATTR_OP_MAX - 1)
enum {
  CTRL_ATTR_MCAST_GRP_UNSPEC,
  CTRL_ATTR_MCAST_GRP_NAME,
  CTRL_ATTR_MCAST_GRP_ID,
  __CTRL_ATTR_MCAST_GRP_MAX,
};
#define CTRL_ATTR_MCAST_GRP_MAX (__CTRL_ATTR_MCAST_GRP_MAX - 1)
enum {
  CTRL_ATTR_POLICY_UNSPEC,
  CTRL_ATTR_POLICY_DO,
  CTRL_ATTR_POLICY_DUMP,
  __CTRL_ATTR_POLICY_DUMP_MAX,
  CTRL_ATTR_POLICY_DUMP_MAX = __CTRL_ATTR_POLICY_DUMP_MAX - 1
};
#define CTRL_ATTR_POLICY_MAX (__CTRL_ATTR_POLICY_DUMP_MAX - 1)
#endif
