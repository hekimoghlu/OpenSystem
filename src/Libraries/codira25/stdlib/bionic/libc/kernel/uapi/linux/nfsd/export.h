/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
#ifndef _UAPINFSD_EXPORT_H
#define _UAPINFSD_EXPORT_H
#include <linux/types.h>
#define NFSCLNT_IDMAX 1024
#define NFSCLNT_ADDRMAX 16
#define NFSCLNT_KEYMAX 32
#define NFSEXP_READONLY 0x0001
#define NFSEXP_INSECURE_PORT 0x0002
#define NFSEXP_ROOTSQUASH 0x0004
#define NFSEXP_ALLSQUASH 0x0008
#define NFSEXP_ASYNC 0x0010
#define NFSEXP_GATHERED_WRITES 0x0020
#define NFSEXP_NOREADDIRPLUS 0x0040
#define NFSEXP_SECURITY_LABEL 0x0080
#define NFSEXP_NOHIDE 0x0200
#define NFSEXP_NOSUBTREECHECK 0x0400
#define NFSEXP_NOAUTHNLM 0x0800
#define NFSEXP_MSNFS 0x1000
#define NFSEXP_FSID 0x2000
#define NFSEXP_CROSSMOUNT 0x4000
#define NFSEXP_NOACL 0x8000
#define NFSEXP_V4ROOT 0x10000
#define NFSEXP_PNFS 0x20000
#define NFSEXP_ALLFLAGS 0x3FEFF
#define NFSEXP_SECINFO_FLAGS (NFSEXP_READONLY | NFSEXP_ROOTSQUASH | NFSEXP_ALLSQUASH | NFSEXP_INSECURE_PORT)
#define NFSEXP_XPRTSEC_NONE 0x0001
#define NFSEXP_XPRTSEC_TLS 0x0002
#define NFSEXP_XPRTSEC_MTLS 0x0004
#define NFSEXP_XPRTSEC_NUM (3)
#define NFSEXP_XPRTSEC_ALL (NFSEXP_XPRTSEC_NONE | NFSEXP_XPRTSEC_TLS | NFSEXP_XPRTSEC_MTLS)
#endif
