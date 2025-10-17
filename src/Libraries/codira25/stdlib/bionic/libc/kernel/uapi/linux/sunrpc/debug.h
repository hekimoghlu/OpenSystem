/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
#ifndef _UAPI_LINUX_SUNRPC_DEBUG_H_
#define _UAPI_LINUX_SUNRPC_DEBUG_H_
#define RPCDBG_XPRT 0x0001
#define RPCDBG_CALL 0x0002
#define RPCDBG_DEBUG 0x0004
#define RPCDBG_NFS 0x0008
#define RPCDBG_AUTH 0x0010
#define RPCDBG_BIND 0x0020
#define RPCDBG_SCHED 0x0040
#define RPCDBG_TRANS 0x0080
#define RPCDBG_SVCXPRT 0x0100
#define RPCDBG_SVCDSP 0x0200
#define RPCDBG_MISC 0x0400
#define RPCDBG_CACHE 0x0800
#define RPCDBG_ALL 0x7fff
enum {
  CTL_RPCDEBUG = 1,
  CTL_NFSDEBUG,
  CTL_NFSDDEBUG,
  CTL_NLMDEBUG,
  CTL_SLOTTABLE_UDP,
  CTL_SLOTTABLE_TCP,
  CTL_MIN_RESVPORT,
  CTL_MAX_RESVPORT,
};
#endif
