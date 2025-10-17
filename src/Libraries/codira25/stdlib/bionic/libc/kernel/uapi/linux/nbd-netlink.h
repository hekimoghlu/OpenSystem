/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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
#ifndef _UAPILINUX_NBD_NETLINK_H
#define _UAPILINUX_NBD_NETLINK_H
#define NBD_GENL_FAMILY_NAME "nbd"
#define NBD_GENL_VERSION 0x1
#define NBD_GENL_MCAST_GROUP_NAME "nbd_mc_group"
enum {
  NBD_ATTR_UNSPEC,
  NBD_ATTR_INDEX,
  NBD_ATTR_SIZE_BYTES,
  NBD_ATTR_BLOCK_SIZE_BYTES,
  NBD_ATTR_TIMEOUT,
  NBD_ATTR_SERVER_FLAGS,
  NBD_ATTR_CLIENT_FLAGS,
  NBD_ATTR_SOCKETS,
  NBD_ATTR_DEAD_CONN_TIMEOUT,
  NBD_ATTR_DEVICE_LIST,
  NBD_ATTR_BACKEND_IDENTIFIER,
  __NBD_ATTR_MAX,
};
#define NBD_ATTR_MAX (__NBD_ATTR_MAX - 1)
enum {
  NBD_DEVICE_ITEM_UNSPEC,
  NBD_DEVICE_ITEM,
  __NBD_DEVICE_ITEM_MAX,
};
#define NBD_DEVICE_ITEM_MAX (__NBD_DEVICE_ITEM_MAX - 1)
enum {
  NBD_DEVICE_UNSPEC,
  NBD_DEVICE_INDEX,
  NBD_DEVICE_CONNECTED,
  __NBD_DEVICE_MAX,
};
#define NBD_DEVICE_ATTR_MAX (__NBD_DEVICE_MAX - 1)
enum {
  NBD_SOCK_ITEM_UNSPEC,
  NBD_SOCK_ITEM,
  __NBD_SOCK_ITEM_MAX,
};
#define NBD_SOCK_ITEM_MAX (__NBD_SOCK_ITEM_MAX - 1)
enum {
  NBD_SOCK_UNSPEC,
  NBD_SOCK_FD,
  __NBD_SOCK_MAX,
};
#define NBD_SOCK_MAX (__NBD_SOCK_MAX - 1)
enum {
  NBD_CMD_UNSPEC,
  NBD_CMD_CONNECT,
  NBD_CMD_DISCONNECT,
  NBD_CMD_RECONFIGURE,
  NBD_CMD_LINK_DEAD,
  NBD_CMD_STATUS,
  __NBD_CMD_MAX,
};
#define NBD_CMD_MAX (__NBD_CMD_MAX - 1)
#endif
