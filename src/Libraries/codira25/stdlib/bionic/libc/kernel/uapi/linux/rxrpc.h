/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 11, 2023.
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
#ifndef _UAPI_LINUX_RXRPC_H
#define _UAPI_LINUX_RXRPC_H
#include <linux/types.h>
#include <linux/in.h>
#include <linux/in6.h>
struct sockaddr_rxrpc {
  __kernel_sa_family_t srx_family;
  __u16 srx_service;
  __u16 transport_type;
  __u16 transport_len;
  union {
    __kernel_sa_family_t family;
    struct sockaddr_in sin;
    struct sockaddr_in6 sin6;
  } transport;
};
#define RXRPC_SECURITY_KEY 1
#define RXRPC_SECURITY_KEYRING 2
#define RXRPC_EXCLUSIVE_CONNECTION 3
#define RXRPC_MIN_SECURITY_LEVEL 4
#define RXRPC_UPGRADEABLE_SERVICE 5
#define RXRPC_SUPPORTED_CMSG 6
enum rxrpc_cmsg_type {
  RXRPC_USER_CALL_ID = 1,
  RXRPC_ABORT = 2,
  RXRPC_ACK = 3,
  RXRPC_NET_ERROR = 5,
  RXRPC_BUSY = 6,
  RXRPC_LOCAL_ERROR = 7,
  RXRPC_NEW_CALL = 8,
  RXRPC_EXCLUSIVE_CALL = 10,
  RXRPC_UPGRADE_SERVICE = 11,
  RXRPC_TX_LENGTH = 12,
  RXRPC_SET_CALL_TIMEOUT = 13,
  RXRPC_CHARGE_ACCEPT = 14,
  RXRPC__SUPPORTED
};
#define RXRPC_SECURITY_PLAIN 0
#define RXRPC_SECURITY_AUTH 1
#define RXRPC_SECURITY_ENCRYPT 2
#define RXRPC_SECURITY_NONE 0
#define RXRPC_SECURITY_RXKAD 2
#define RXRPC_SECURITY_RXGK 4
#define RXRPC_SECURITY_RXK5 5
#define RX_CALL_DEAD - 1
#define RX_INVALID_OPERATION - 2
#define RX_CALL_TIMEOUT - 3
#define RX_EOF - 4
#define RX_PROTOCOL_ERROR - 5
#define RX_USER_ABORT - 6
#define RX_ADDRINUSE - 7
#define RX_DEBUGI_BADTYPE - 8
#define RXGEN_CC_MARSHAL - 450
#define RXGEN_CC_UNMARSHAL - 451
#define RXGEN_SS_MARSHAL - 452
#define RXGEN_SS_UNMARSHAL - 453
#define RXGEN_DECODE - 454
#define RXGEN_OPCODE - 455
#define RXGEN_SS_XDRFREE - 456
#define RXGEN_CC_XDRFREE - 457
#define RXKADINCONSISTENCY 19270400
#define RXKADPACKETSHORT 19270401
#define RXKADLEVELFAIL 19270402
#define RXKADTICKETLEN 19270403
#define RXKADOUTOFSEQUENCE 19270404
#define RXKADNOAUTH 19270405
#define RXKADBADKEY 19270406
#define RXKADBADTICKET 19270407
#define RXKADUNKNOWNKEY 19270408
#define RXKADEXPIRED 19270409
#define RXKADSEALEDINCON 19270410
#define RXKADDATALEN 19270411
#define RXKADILLEGALLEVEL 19270412
#endif
