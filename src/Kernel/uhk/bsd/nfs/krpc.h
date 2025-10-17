/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 7, 2021.
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
/* Copyright (c) 1995 NeXT Computer, Inc. All Rights Reserved */

#ifndef __NFS_KRPC_H__
#define __NFS_KRPC_H__

#include <sys/appleapiopts.h>

#include <sys/cdefs.h>

#ifdef __APPLE_API_PRIVATE

/*
 * RPC definitions for the portmapper (portmap and rpcbind)
 */
#define PMAPPORT                111
#define PMAPPROG                100000
#define PMAPVERS                2
#define PMAPPROC_NULL           0
#define PMAPPROC_SET            1
#define PMAPPROC_UNSET          2
#define PMAPPROC_GETPORT        3
#define PMAPPROC_DUMP           4
#define PMAPPROC_CALLIT         5

#define RPCBPROG                PMAPPROG
#define RPCBVERS3               3
#define RPCBVERS4               4
#define RPCBPROC_NULL           0
#define RPCBPROC_SET            1
#define RPCBPROC_UNSET          2
#define RPCBPROC_GETADDR        3
#define RPCBPROC_DUMP           4
#define RPCBPROC_CALLIT         5
#define RPCBPROC_BCAST          RPCBPROC_CALLIT
#define RPCBPROC_GETTIME        6
#define RPCBPROC_UADDR2TADDR    7
#define RPCBPROC_TADDR2UADDR    8
#define RPCBPROC_GETVERSADDR    9
#define RPCBPROC_INDIRECT       10
#define RPCBPROC_GETADDRLIST    11
#define RPCBPROC_GETSTAT        12


/*
 * RPC definitions for bootparamd
 */
#define BOOTPARAM_PROG          100026
#define BOOTPARAM_VERS          1
#define BOOTPARAM_WHOAMI        1
#define BOOTPARAM_GETFILE       2

#endif /* __APPLE_API_PRIVATE */
#endif /* __NFS_KRPC_H__ */
