/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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
 * IPv6Sock_Compat.h
 * - shim file for facilitating the transition of RFC 2292 to RFC 3542 
 *   socket options
 */
/* 
 * Modification History
 *
 * February 1, 2011		dX (wxie@apple.com)
 * - initial version
 */

#ifndef _S_IPV6SOCKCOMPAT_H
#define _S_IPV6SOCKCOMPAT_H

#include <netinet/in.h>

#ifdef IPV6_RECVPKTINFO
#define IPCONFIG_SOCKOPT_PKTINFO IPV6_RECVPKTINFO
#else
#define IPCONFIG_SOCKOPT_PKTINFO IPV6_PKTINFO
#endif

#ifdef IPV6_RECVHOPLIMIT
#define IPCONFIG_SOCKOPT_HOPLIMIT IPV6_RECVHOPLIMIT
#else
#define IPCONFIG_SOCKOPT_HOPLIMIT IPV6_HOPLIMIT 
#endif


#endif /* _S_IPV6SOCKCOMPAT_H */
