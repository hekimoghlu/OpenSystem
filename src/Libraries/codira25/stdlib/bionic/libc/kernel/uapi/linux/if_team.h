/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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
#ifndef _UAPI_LINUX_IF_TEAM_H
#define _UAPI_LINUX_IF_TEAM_H
#define TEAM_GENL_NAME "team"
#define TEAM_GENL_VERSION 1
#define TEAM_STRING_MAX_LEN 32
#define TEAM_GENL_CHANGE_EVENT_MC_GRP_NAME "change_event"
enum {
  TEAM_ATTR_UNSPEC,
  TEAM_ATTR_TEAM_IFINDEX,
  TEAM_ATTR_LIST_OPTION,
  TEAM_ATTR_LIST_PORT,
  __TEAM_ATTR_MAX,
  TEAM_ATTR_MAX = (__TEAM_ATTR_MAX - 1)
};
enum {
  TEAM_ATTR_ITEM_OPTION_UNSPEC,
  TEAM_ATTR_ITEM_OPTION,
  __TEAM_ATTR_ITEM_OPTION_MAX,
  TEAM_ATTR_ITEM_OPTION_MAX = (__TEAM_ATTR_ITEM_OPTION_MAX - 1)
};
enum {
  TEAM_ATTR_OPTION_UNSPEC,
  TEAM_ATTR_OPTION_NAME,
  TEAM_ATTR_OPTION_CHANGED,
  TEAM_ATTR_OPTION_TYPE,
  TEAM_ATTR_OPTION_DATA,
  TEAM_ATTR_OPTION_REMOVED,
  TEAM_ATTR_OPTION_PORT_IFINDEX,
  TEAM_ATTR_OPTION_ARRAY_INDEX,
  __TEAM_ATTR_OPTION_MAX,
  TEAM_ATTR_OPTION_MAX = (__TEAM_ATTR_OPTION_MAX - 1)
};
enum {
  TEAM_ATTR_ITEM_PORT_UNSPEC,
  TEAM_ATTR_ITEM_PORT,
  __TEAM_ATTR_ITEM_PORT_MAX,
  TEAM_ATTR_ITEM_PORT_MAX = (__TEAM_ATTR_ITEM_PORT_MAX - 1)
};
enum {
  TEAM_ATTR_PORT_UNSPEC,
  TEAM_ATTR_PORT_IFINDEX,
  TEAM_ATTR_PORT_CHANGED,
  TEAM_ATTR_PORT_LINKUP,
  TEAM_ATTR_PORT_SPEED,
  TEAM_ATTR_PORT_DUPLEX,
  TEAM_ATTR_PORT_REMOVED,
  __TEAM_ATTR_PORT_MAX,
  TEAM_ATTR_PORT_MAX = (__TEAM_ATTR_PORT_MAX - 1)
};
enum {
  TEAM_CMD_NOOP,
  TEAM_CMD_OPTIONS_SET,
  TEAM_CMD_OPTIONS_GET,
  TEAM_CMD_PORT_LIST_GET,
  __TEAM_CMD_MAX,
  TEAM_CMD_MAX = (__TEAM_CMD_MAX - 1)
};
#endif
