/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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
#ifndef __UAPI_NCSI_NETLINK_H__
#define __UAPI_NCSI_NETLINK_H__
enum ncsi_nl_commands {
  NCSI_CMD_UNSPEC,
  NCSI_CMD_PKG_INFO,
  NCSI_CMD_SET_INTERFACE,
  NCSI_CMD_CLEAR_INTERFACE,
  NCSI_CMD_SEND_CMD,
  NCSI_CMD_SET_PACKAGE_MASK,
  NCSI_CMD_SET_CHANNEL_MASK,
  __NCSI_CMD_AFTER_LAST,
  NCSI_CMD_MAX = __NCSI_CMD_AFTER_LAST - 1
};
enum ncsi_nl_attrs {
  NCSI_ATTR_UNSPEC,
  NCSI_ATTR_IFINDEX,
  NCSI_ATTR_PACKAGE_LIST,
  NCSI_ATTR_PACKAGE_ID,
  NCSI_ATTR_CHANNEL_ID,
  NCSI_ATTR_DATA,
  NCSI_ATTR_MULTI_FLAG,
  NCSI_ATTR_PACKAGE_MASK,
  NCSI_ATTR_CHANNEL_MASK,
  __NCSI_ATTR_AFTER_LAST,
  NCSI_ATTR_MAX = __NCSI_ATTR_AFTER_LAST - 1
};
enum ncsi_nl_pkg_attrs {
  NCSI_PKG_ATTR_UNSPEC,
  NCSI_PKG_ATTR,
  NCSI_PKG_ATTR_ID,
  NCSI_PKG_ATTR_FORCED,
  NCSI_PKG_ATTR_CHANNEL_LIST,
  __NCSI_PKG_ATTR_AFTER_LAST,
  NCSI_PKG_ATTR_MAX = __NCSI_PKG_ATTR_AFTER_LAST - 1
};
enum ncsi_nl_channel_attrs {
  NCSI_CHANNEL_ATTR_UNSPEC,
  NCSI_CHANNEL_ATTR,
  NCSI_CHANNEL_ATTR_ID,
  NCSI_CHANNEL_ATTR_VERSION_MAJOR,
  NCSI_CHANNEL_ATTR_VERSION_MINOR,
  NCSI_CHANNEL_ATTR_VERSION_STR,
  NCSI_CHANNEL_ATTR_LINK_STATE,
  NCSI_CHANNEL_ATTR_ACTIVE,
  NCSI_CHANNEL_ATTR_FORCED,
  NCSI_CHANNEL_ATTR_VLAN_LIST,
  NCSI_CHANNEL_ATTR_VLAN_ID,
  __NCSI_CHANNEL_ATTR_AFTER_LAST,
  NCSI_CHANNEL_ATTR_MAX = __NCSI_CHANNEL_ATTR_AFTER_LAST - 1
};
#endif
