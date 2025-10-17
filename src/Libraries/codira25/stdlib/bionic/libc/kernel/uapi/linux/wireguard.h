/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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
#ifndef _WG_UAPI_WIREGUARD_H
#define _WG_UAPI_WIREGUARD_H
#define WG_GENL_NAME "wireguard"
#define WG_GENL_VERSION 1
#define WG_KEY_LEN 32
enum wg_cmd {
  WG_CMD_GET_DEVICE,
  WG_CMD_SET_DEVICE,
  __WG_CMD_MAX
};
#define WG_CMD_MAX (__WG_CMD_MAX - 1)
enum wgdevice_flag {
  WGDEVICE_F_REPLACE_PEERS = 1U << 0,
  __WGDEVICE_F_ALL = WGDEVICE_F_REPLACE_PEERS
};
enum wgdevice_attribute {
  WGDEVICE_A_UNSPEC,
  WGDEVICE_A_IFINDEX,
  WGDEVICE_A_IFNAME,
  WGDEVICE_A_PRIVATE_KEY,
  WGDEVICE_A_PUBLIC_KEY,
  WGDEVICE_A_FLAGS,
  WGDEVICE_A_LISTEN_PORT,
  WGDEVICE_A_FWMARK,
  WGDEVICE_A_PEERS,
  __WGDEVICE_A_LAST
};
#define WGDEVICE_A_MAX (__WGDEVICE_A_LAST - 1)
enum wgpeer_flag {
  WGPEER_F_REMOVE_ME = 1U << 0,
  WGPEER_F_REPLACE_ALLOWEDIPS = 1U << 1,
  WGPEER_F_UPDATE_ONLY = 1U << 2,
  __WGPEER_F_ALL = WGPEER_F_REMOVE_ME | WGPEER_F_REPLACE_ALLOWEDIPS | WGPEER_F_UPDATE_ONLY
};
enum wgpeer_attribute {
  WGPEER_A_UNSPEC,
  WGPEER_A_PUBLIC_KEY,
  WGPEER_A_PRESHARED_KEY,
  WGPEER_A_FLAGS,
  WGPEER_A_ENDPOINT,
  WGPEER_A_PERSISTENT_KEEPALIVE_INTERVAL,
  WGPEER_A_LAST_HANDSHAKE_TIME,
  WGPEER_A_RX_BYTES,
  WGPEER_A_TX_BYTES,
  WGPEER_A_ALLOWEDIPS,
  WGPEER_A_PROTOCOL_VERSION,
  __WGPEER_A_LAST
};
#define WGPEER_A_MAX (__WGPEER_A_LAST - 1)
enum wgallowedip_attribute {
  WGALLOWEDIP_A_UNSPEC,
  WGALLOWEDIP_A_FAMILY,
  WGALLOWEDIP_A_IPADDR,
  WGALLOWEDIP_A_CIDR_MASK,
  __WGALLOWEDIP_A_LAST
};
#define WGALLOWEDIP_A_MAX (__WGALLOWEDIP_A_LAST - 1)
#endif
