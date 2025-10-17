/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
#ifndef _UAPI_LINUX_NETDEVICE_H
#define _UAPI_LINUX_NETDEVICE_H
#include <linux/if.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/if_link.h>
#define MAX_ADDR_LEN 32
#define INIT_NETDEV_GROUP 0
#define NET_NAME_UNKNOWN 0
#define NET_NAME_ENUM 1
#define NET_NAME_PREDICTABLE 2
#define NET_NAME_USER 3
#define NET_NAME_RENAMED 4
enum {
  IF_PORT_UNKNOWN = 0,
  IF_PORT_10BASE2,
  IF_PORT_10BASET,
  IF_PORT_AUI,
  IF_PORT_100BASET,
  IF_PORT_100BASETX,
  IF_PORT_100BASEFX
};
#define NET_ADDR_PERM 0
#define NET_ADDR_RANDOM 1
#define NET_ADDR_STOLEN 2
#define NET_ADDR_SET 3
#endif
