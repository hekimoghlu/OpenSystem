/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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
/*
 * Options of interest only in fstab entries.
 */
#define FSTAB_MNT_NET   0x00000001
#define MOPT_NET        { "net",	0, FSTAB_MNT_NET, 1 }

/*
 * NFS-specific mount options.
 */
#define NFS_MNT_PORT    0x00000001
#define NFS_MNT_VERS    0x00000002
#define NFS_MNT_NFSVERS 0x00000004
#define NFS_MNT_PROTO   0x00000008
#define NFS_MNT_TCP     0x00000010
#define NFS_MNT_UDP     0x00000020

#define MOPT_VERS \
	{ "vers",	0, NFS_MNT_VERS, 1 }, \
	{ "nfsvers",	0, NFS_MNT_NFSVERS, 1 }

#define MOPT_NFS \
	{ "port",	0, NFS_MNT_PORT, 1 },   \
	MOPT_VERS,                              \
	{ "proto",	0, NFS_MNT_PROTO, 1 },  \
	{ "tcp",	0, NFS_MNT_TCP, 1 },    \
	{ "udp",	0, NFS_MNT_UDP, 1 },    \
	{ NULL,		0, 0, 0 }

extern int get_nfs_vers(mntoptparse_t, int);
