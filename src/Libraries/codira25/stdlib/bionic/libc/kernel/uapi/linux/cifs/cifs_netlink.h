/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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
#ifndef _UAPILINUX_CIFS_NETLINK_H
#define _UAPILINUX_CIFS_NETLINK_H
#define CIFS_GENL_NAME "cifs"
#define CIFS_GENL_VERSION 0x1
#define CIFS_GENL_MCGRP_SWN_NAME "cifs_mcgrp_swn"
enum cifs_genl_multicast_groups {
  CIFS_GENL_MCGRP_SWN,
};
enum cifs_genl_attributes {
  CIFS_GENL_ATTR_UNSPEC,
  CIFS_GENL_ATTR_SWN_REGISTRATION_ID,
  CIFS_GENL_ATTR_SWN_NET_NAME,
  CIFS_GENL_ATTR_SWN_SHARE_NAME,
  CIFS_GENL_ATTR_SWN_IP,
  CIFS_GENL_ATTR_SWN_NET_NAME_NOTIFY,
  CIFS_GENL_ATTR_SWN_SHARE_NAME_NOTIFY,
  CIFS_GENL_ATTR_SWN_IP_NOTIFY,
  CIFS_GENL_ATTR_SWN_KRB_AUTH,
  CIFS_GENL_ATTR_SWN_USER_NAME,
  CIFS_GENL_ATTR_SWN_PASSWORD,
  CIFS_GENL_ATTR_SWN_DOMAIN_NAME,
  CIFS_GENL_ATTR_SWN_NOTIFICATION_TYPE,
  CIFS_GENL_ATTR_SWN_RESOURCE_STATE,
  CIFS_GENL_ATTR_SWN_RESOURCE_NAME,
  __CIFS_GENL_ATTR_MAX,
};
#define CIFS_GENL_ATTR_MAX (__CIFS_GENL_ATTR_MAX - 1)
enum cifs_genl_commands {
  CIFS_GENL_CMD_UNSPEC,
  CIFS_GENL_CMD_SWN_REGISTER,
  CIFS_GENL_CMD_SWN_UNREGISTER,
  CIFS_GENL_CMD_SWN_NOTIFY,
  __CIFS_GENL_CMD_MAX
};
#define CIFS_GENL_CMD_MAX (__CIFS_GENL_CMD_MAX - 1)
enum cifs_swn_notification_type {
  CIFS_SWN_NOTIFICATION_RESOURCE_CHANGE = 0x01,
  CIFS_SWN_NOTIFICATION_CLIENT_MOVE = 0x02,
  CIFS_SWN_NOTIFICATION_SHARE_MOVE = 0x03,
  CIFS_SWN_NOTIFICATION_IP_CHANGE = 0x04,
};
enum cifs_swn_resource_state {
  CIFS_SWN_RESOURCE_STATE_UNKNOWN = 0x00,
  CIFS_SWN_RESOURCE_STATE_AVAILABLE = 0x01,
  CIFS_SWN_RESOURCE_STATE_UNAVAILABLE = 0xFF
};
#endif
