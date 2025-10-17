/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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
#ifndef _UAPI_LINUX_SOCKET_H
#define _UAPI_LINUX_SOCKET_H
#include <bits/sockaddr_storage.h>
#define _K_SS_MAXSIZE 128
typedef unsigned short __kernel_sa_family_t;
#define SOCK_SNDBUF_LOCK 1
#define SOCK_RCVBUF_LOCK 2
#define SOCK_BUF_LOCK_MASK (SOCK_SNDBUF_LOCK | SOCK_RCVBUF_LOCK)
#define SOCK_TXREHASH_DEFAULT 255
#define SOCK_TXREHASH_DISABLED 0
#define SOCK_TXREHASH_ENABLED 1
#endif
