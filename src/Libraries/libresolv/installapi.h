/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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

#include <sys/cdefs.h>
#include <mach/mach.h>
#include <netdb.h>
#include <resolv.h>

// These are declared in Libinfo's headers.  These declarations should eventually move into headers provided by libresolv.

typedef void (*dns_async_callback)(int32_t status, char *buf, uint32_t len, struct sockaddr *from, int fromlen, void *context);
int32_t dns_async_start(mach_port_t *p, const char *name, uint16_t dnsclass, uint16_t dnstype, uint32_t do_search, dns_async_callback callback, void *context);
int32_t dns_async_send(mach_port_t *p, const char *name, uint16_t dnsclass, uint16_t dnstype, uint32_t do_search);
int32_t dns_async_receive(mach_port_t p, char **buf, uint32_t *len, struct sockaddr **from, uint32_t *fromlen);
int32_t dns_async_handle_reply(void *msg);
void dns_async_cancel(mach_port_t p);
