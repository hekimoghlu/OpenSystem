/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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
#ifndef _LINUX_QRTR_H
#define _LINUX_QRTR_H
#include <linux/socket.h>
#include <linux/types.h>
#define QRTR_NODE_BCAST 0xffffffffu
#define QRTR_PORT_CTRL 0xfffffffeu
struct sockaddr_qrtr {
  __kernel_sa_family_t sq_family;
  __u32 sq_node;
  __u32 sq_port;
};
enum qrtr_pkt_type {
  QRTR_TYPE_DATA = 1,
  QRTR_TYPE_HELLO = 2,
  QRTR_TYPE_BYE = 3,
  QRTR_TYPE_NEW_SERVER = 4,
  QRTR_TYPE_DEL_SERVER = 5,
  QRTR_TYPE_DEL_CLIENT = 6,
  QRTR_TYPE_RESUME_TX = 7,
  QRTR_TYPE_EXIT = 8,
  QRTR_TYPE_PING = 9,
  QRTR_TYPE_NEW_LOOKUP = 10,
  QRTR_TYPE_DEL_LOOKUP = 11,
};
struct qrtr_ctrl_pkt {
  __le32 cmd;
  union {
    struct {
      __le32 service;
      __le32 instance;
      __le32 node;
      __le32 port;
    } server;
    struct {
      __le32 node;
      __le32 port;
    } client;
  };
} __attribute__((__packed__));
#endif
