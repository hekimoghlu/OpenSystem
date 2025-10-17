/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
#ifndef _RESOLV_NETID_H
#define _RESOLV_NETID_H

/* This header contains declarations related to per-network DNS
 * server selection. They are used by system/netd/ and should not be
 * exposed by the C library's public NDK headers.
 */
#include <sys/cdefs.h>
#include <netinet/in.h>
#include "resolv_params.h"
#include <stdio.h>

/*
 * Passing NETID_UNSET as the netId causes system/netd/server/DnsProxyListener.cpp to
 * fill in the appropriate default netId for the query.
 */
#define NETID_UNSET 0u

/*
 * MARK_UNSET represents the default (i.e. unset) value for a socket mark.
 */
#define MARK_UNSET 0u

__BEGIN_DECLS

struct __res_params;
struct addrinfo;

#define __used_in_netd __attribute__((visibility ("default")))

/*
 * A struct to capture context relevant to network operations.
 *
 * Application and DNS netids/marks can differ from one another under certain
 * circumstances, notably when a VPN applies to the given uid's traffic but the
 * VPN network does not have its own DNS servers explicitly provisioned.
 *
 * The introduction of per-UID routing means the uid is also an essential part
 * of the evaluation context. Its proper uninitialized value is
 * NET_CONTEXT_INVALID_UID.
 */
struct android_net_context {
    unsigned app_netid;
    unsigned app_mark;
    unsigned dns_netid;
    unsigned dns_mark;
    uid_t uid;
    unsigned flags;
    res_send_qhook qhook;
};

#define NET_CONTEXT_INVALID_UID ((uid_t)-1)

#define NET_CONTEXT_FLAG_USE_LOCAL_NAMESERVERS  0x00000001
#define NET_CONTEXT_FLAG_USE_EDNS               0x00000002

struct hostent *android_gethostbyaddrfornet(const void *, socklen_t, int, unsigned, unsigned) __used_in_netd;
struct hostent *android_gethostbynamefornet(const char *, int, unsigned, unsigned) __used_in_netd;
int android_getaddrinfofornet(const char *, const char *, const struct addrinfo *, unsigned,
    unsigned, struct addrinfo **) __used_in_netd;
/*
 * TODO: consider refactoring android_getaddrinfo_proxy() to serve as an
 * explore_fqdn() dispatch table method, with the below function only making DNS calls.
 */
struct hostent *android_gethostbyaddrfornetcontext(const void *, socklen_t, int, const struct android_net_context *) __used_in_netd;
struct hostent *android_gethostbynamefornetcontext(const char *, int, const struct android_net_context *) __used_in_netd;
int android_getaddrinfofornetcontext(const char *, const char *, const struct addrinfo *,
    const struct android_net_context *, struct addrinfo **) __used_in_netd;

/* set name servers for a network */
extern int _resolv_set_nameservers_for_net(unsigned netid, const char** servers,
        unsigned numservers, const char *domains, const struct __res_params* params) __used_in_netd;

/* flush the cache associated with a certain network */
extern void _resolv_flush_cache_for_net(unsigned netid) __used_in_netd;

/* delete the cache associated with a certain network */
extern void _resolv_delete_cache_for_net(unsigned netid) __used_in_netd;

/* Internal use only. */
struct hostent *android_gethostbyaddrfornetcontext_proxy(const void *, socklen_t, int , const struct android_net_context *) __LIBC_HIDDEN__;
int android_getnameinfofornet(const struct sockaddr *, socklen_t, char *, size_t, char *, size_t, int, unsigned, unsigned) __LIBC_HIDDEN__;

__END_DECLS

#endif /* _RESOLV_NETID_H */
