/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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
#ifndef _UAPI_VM_SOCKETS_H
#define _UAPI_VM_SOCKETS_H
#include <linux/socket.h>
#include <linux/types.h>
#define SO_VM_SOCKETS_BUFFER_SIZE 0
#define SO_VM_SOCKETS_BUFFER_MIN_SIZE 1
#define SO_VM_SOCKETS_BUFFER_MAX_SIZE 2
#define SO_VM_SOCKETS_PEER_HOST_VM_ID 3
#define SO_VM_SOCKETS_TRUSTED 5
#define SO_VM_SOCKETS_CONNECT_TIMEOUT_OLD 6
#define SO_VM_SOCKETS_NONBLOCK_TXRX 7
#define SO_VM_SOCKETS_CONNECT_TIMEOUT_NEW 8
#if __BITS_PER_LONG == 64 || defined(__x86_64__) && defined(__ILP32__)
#define SO_VM_SOCKETS_CONNECT_TIMEOUT SO_VM_SOCKETS_CONNECT_TIMEOUT_OLD
#else
#define SO_VM_SOCKETS_CONNECT_TIMEOUT (sizeof(time_t) == sizeof(__kernel_long_t) ? SO_VM_SOCKETS_CONNECT_TIMEOUT_OLD : SO_VM_SOCKETS_CONNECT_TIMEOUT_NEW)
#endif
#define VMADDR_CID_ANY - 1U
#define VMADDR_PORT_ANY - 1U
#define VMADDR_CID_HYPERVISOR 0
#define VMADDR_CID_LOCAL 1
#define VMADDR_CID_HOST 2
#define VMADDR_FLAG_TO_HOST 0x01
#define VM_SOCKETS_INVALID_VERSION - 1U
#define VM_SOCKETS_VERSION_EPOCH(_v) (((_v) & 0xFF000000) >> 24)
#define VM_SOCKETS_VERSION_MAJOR(_v) (((_v) & 0x00FF0000) >> 16)
#define VM_SOCKETS_VERSION_MINOR(_v) (((_v) & 0x0000FFFF))
struct sockaddr_vm {
  __kernel_sa_family_t svm_family;
  unsigned short svm_reserved1;
  unsigned int svm_port;
  unsigned int svm_cid;
  __u8 svm_flags;
  unsigned char svm_zero[sizeof(struct sockaddr) - sizeof(sa_family_t) - sizeof(unsigned short) - sizeof(unsigned int) - sizeof(unsigned int) - sizeof(__u8)];
};
#define IOCTL_VM_SOCKETS_GET_LOCAL_CID _IO(7, 0xb9)
#define SOL_VSOCK 287
#define VSOCK_RECVERR 1
#endif
