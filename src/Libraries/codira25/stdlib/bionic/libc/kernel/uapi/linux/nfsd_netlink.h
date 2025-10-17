/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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
#ifndef _UAPI_LINUX_NFSD_NETLINK_H
#define _UAPI_LINUX_NFSD_NETLINK_H
#define NFSD_FAMILY_NAME "nfsd"
#define NFSD_FAMILY_VERSION 1
enum {
  NFSD_A_RPC_STATUS_XID = 1,
  NFSD_A_RPC_STATUS_FLAGS,
  NFSD_A_RPC_STATUS_PROG,
  NFSD_A_RPC_STATUS_VERSION,
  NFSD_A_RPC_STATUS_PROC,
  NFSD_A_RPC_STATUS_SERVICE_TIME,
  NFSD_A_RPC_STATUS_PAD,
  NFSD_A_RPC_STATUS_SADDR4,
  NFSD_A_RPC_STATUS_DADDR4,
  NFSD_A_RPC_STATUS_SADDR6,
  NFSD_A_RPC_STATUS_DADDR6,
  NFSD_A_RPC_STATUS_SPORT,
  NFSD_A_RPC_STATUS_DPORT,
  NFSD_A_RPC_STATUS_COMPOUND_OPS,
  __NFSD_A_RPC_STATUS_MAX,
  NFSD_A_RPC_STATUS_MAX = (__NFSD_A_RPC_STATUS_MAX - 1)
};
enum {
  NFSD_A_SERVER_THREADS = 1,
  NFSD_A_SERVER_GRACETIME,
  NFSD_A_SERVER_LEASETIME,
  NFSD_A_SERVER_SCOPE,
  __NFSD_A_SERVER_MAX,
  NFSD_A_SERVER_MAX = (__NFSD_A_SERVER_MAX - 1)
};
enum {
  NFSD_A_VERSION_MAJOR = 1,
  NFSD_A_VERSION_MINOR,
  NFSD_A_VERSION_ENABLED,
  __NFSD_A_VERSION_MAX,
  NFSD_A_VERSION_MAX = (__NFSD_A_VERSION_MAX - 1)
};
enum {
  NFSD_A_SERVER_PROTO_VERSION = 1,
  __NFSD_A_SERVER_PROTO_MAX,
  NFSD_A_SERVER_PROTO_MAX = (__NFSD_A_SERVER_PROTO_MAX - 1)
};
enum {
  NFSD_A_SOCK_ADDR = 1,
  NFSD_A_SOCK_TRANSPORT_NAME,
  __NFSD_A_SOCK_MAX,
  NFSD_A_SOCK_MAX = (__NFSD_A_SOCK_MAX - 1)
};
enum {
  NFSD_A_SERVER_SOCK_ADDR = 1,
  __NFSD_A_SERVER_SOCK_MAX,
  NFSD_A_SERVER_SOCK_MAX = (__NFSD_A_SERVER_SOCK_MAX - 1)
};
enum {
  NFSD_A_POOL_MODE_MODE = 1,
  NFSD_A_POOL_MODE_NPOOLS,
  __NFSD_A_POOL_MODE_MAX,
  NFSD_A_POOL_MODE_MAX = (__NFSD_A_POOL_MODE_MAX - 1)
};
enum {
  NFSD_CMD_RPC_STATUS_GET = 1,
  NFSD_CMD_THREADS_SET,
  NFSD_CMD_THREADS_GET,
  NFSD_CMD_VERSION_SET,
  NFSD_CMD_VERSION_GET,
  NFSD_CMD_LISTENER_SET,
  NFSD_CMD_LISTENER_GET,
  NFSD_CMD_POOL_MODE_SET,
  NFSD_CMD_POOL_MODE_GET,
  __NFSD_CMD_MAX,
  NFSD_CMD_MAX = (__NFSD_CMD_MAX - 1)
};
#endif
