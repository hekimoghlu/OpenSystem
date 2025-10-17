/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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
#ifndef __DNS_PROXY_H
#define __DNS_PROXY_H

#include "mDNSEmbeddedAPI.h"
#include "DNSCommon.h"

extern void ProxyUDPCallback(void *socket, DNSMessage *const msg, const mDNSu8 *const end, const mDNSAddr *const srcaddr,
                             const mDNSIPPort srcport, const mDNSAddr *dstaddr, const mDNSIPPort dstport, const mDNSInterfaceID InterfaceID, void *context);
extern void ProxyTCPCallback(void *socket, DNSMessage *const msg, const mDNSu8 *const end, const mDNSAddr *const srcaddr,
                             const mDNSIPPort srcport, const mDNSAddr *dstaddr, const mDNSIPPort dstport, const mDNSInterfaceID InterfaceID, void *context);

extern const struct mrcs_server_dns_proxy_handlers_s kMRCSServerDNSProxyHandlers;

#endif // __DNS_PROXY_H
