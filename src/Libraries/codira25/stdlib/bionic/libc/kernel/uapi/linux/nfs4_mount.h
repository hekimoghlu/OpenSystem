/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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
#ifndef _LINUX_NFS4_MOUNT_H
#define _LINUX_NFS4_MOUNT_H
#define NFS4_MOUNT_VERSION 1
struct nfs_string {
  unsigned int len;
  const char  * data;
};
struct nfs4_mount_data {
  int version;
  int flags;
  int rsize;
  int wsize;
  int timeo;
  int retrans;
  int acregmin;
  int acregmax;
  int acdirmin;
  int acdirmax;
  struct nfs_string client_addr;
  struct nfs_string mnt_path;
  struct nfs_string hostname;
  unsigned int host_addrlen;
  struct sockaddr  * host_addr;
  int proto;
  int auth_flavourlen;
  int  * auth_flavours;
};
#define NFS4_MOUNT_SOFT 0x0001
#define NFS4_MOUNT_INTR 0x0002
#define NFS4_MOUNT_NOCTO 0x0010
#define NFS4_MOUNT_NOAC 0x0020
#define NFS4_MOUNT_STRICTLOCK 0x1000
#define NFS4_MOUNT_UNSHARED 0x8000
#define NFS4_MOUNT_FLAGMASK 0x9033
#endif
