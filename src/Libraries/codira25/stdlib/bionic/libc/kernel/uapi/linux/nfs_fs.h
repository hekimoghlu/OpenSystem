/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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
#ifndef _UAPI_LINUX_NFS_FS_H
#define _UAPI_LINUX_NFS_FS_H
#include <linux/magic.h>
#define NFS_DEF_UDP_TIMEO (11)
#define NFS_DEF_UDP_RETRANS (3)
#define NFS_DEF_TCP_TIMEO (600)
#define NFS_DEF_TCP_RETRANS (2)
#define NFS_MAX_UDP_TIMEOUT (60 * HZ)
#define NFS_MAX_TCP_TIMEOUT (600 * HZ)
#define NFS_DEF_ACREGMIN (3)
#define NFS_DEF_ACREGMAX (60)
#define NFS_DEF_ACDIRMIN (30)
#define NFS_DEF_ACDIRMAX (60)
#define FLUSH_SYNC 1
#define FLUSH_STABLE 4
#define FLUSH_LOWPRI 8
#define FLUSH_HIGHPRI 16
#define FLUSH_COND_STABLE 32
#define NFSDBG_VFS 0x0001
#define NFSDBG_DIRCACHE 0x0002
#define NFSDBG_LOOKUPCACHE 0x0004
#define NFSDBG_PAGECACHE 0x0008
#define NFSDBG_PROC 0x0010
#define NFSDBG_XDR 0x0020
#define NFSDBG_FILE 0x0040
#define NFSDBG_ROOT 0x0080
#define NFSDBG_CALLBACK 0x0100
#define NFSDBG_CLIENT 0x0200
#define NFSDBG_MOUNT 0x0400
#define NFSDBG_FSCACHE 0x0800
#define NFSDBG_PNFS 0x1000
#define NFSDBG_PNFS_LD 0x2000
#define NFSDBG_STATE 0x4000
#define NFSDBG_XATTRCACHE 0x8000
#define NFSDBG_ALL 0xFFFF
#endif
