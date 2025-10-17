/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 1, 2025.
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
#ifndef _LINUX_NFS2_H
#define _LINUX_NFS2_H
#define NFS2_PORT 2049
#define NFS2_MAXDATA 8192
#define NFS2_MAXPATHLEN 1024
#define NFS2_MAXNAMLEN 255
#define NFS2_MAXGROUPS 16
#define NFS2_FHSIZE 32
#define NFS2_COOKIESIZE 4
#define NFS2_FIFO_DEV (- 1)
#define NFS2MODE_FMT 0170000
#define NFS2MODE_DIR 0040000
#define NFS2MODE_CHR 0020000
#define NFS2MODE_BLK 0060000
#define NFS2MODE_REG 0100000
#define NFS2MODE_LNK 0120000
#define NFS2MODE_SOCK 0140000
#define NFS2MODE_FIFO 0010000
enum nfs2_ftype {
  NF2NON = 0,
  NF2REG = 1,
  NF2DIR = 2,
  NF2BLK = 3,
  NF2CHR = 4,
  NF2LNK = 5,
  NF2SOCK = 6,
  NF2BAD = 7,
  NF2FIFO = 8
};
struct nfs2_fh {
  char data[NFS2_FHSIZE];
};
#define NFS2_VERSION 2
#define NFSPROC_NULL 0
#define NFSPROC_GETATTR 1
#define NFSPROC_SETATTR 2
#define NFSPROC_ROOT 3
#define NFSPROC_LOOKUP 4
#define NFSPROC_READLINK 5
#define NFSPROC_READ 6
#define NFSPROC_WRITECACHE 7
#define NFSPROC_WRITE 8
#define NFSPROC_CREATE 9
#define NFSPROC_REMOVE 10
#define NFSPROC_RENAME 11
#define NFSPROC_LINK 12
#define NFSPROC_SYMLINK 13
#define NFSPROC_MKDIR 14
#define NFSPROC_RMDIR 15
#define NFSPROC_READDIR 16
#define NFSPROC_STATFS 17
#endif
