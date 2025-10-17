/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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
#ifndef _UAPILINUX_NFSD_DEBUG_H
#define _UAPILINUX_NFSD_DEBUG_H
#include <linux/sunrpc/debug.h>
#define NFSDDBG_SOCK 0x0001
#define NFSDDBG_FH 0x0002
#define NFSDDBG_EXPORT 0x0004
#define NFSDDBG_SVC 0x0008
#define NFSDDBG_PROC 0x0010
#define NFSDDBG_FILEOP 0x0020
#define NFSDDBG_AUTH 0x0040
#define NFSDDBG_REPCACHE 0x0080
#define NFSDDBG_XDR 0x0100
#define NFSDDBG_LOCKD 0x0200
#define NFSDDBG_PNFS 0x0400
#define NFSDDBG_ALL 0x7FFF
#define NFSDDBG_NOCHANGE 0xFFFF
#endif
