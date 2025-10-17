/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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
#ifndef SUDOERS_INTERFACES_H
#define SUDOERS_INTERFACES_H

/*
 * Union to hold either strucr in_addr or in6_add
 */
union sudo_in_addr_un {
    struct in_addr ip4;
#ifdef HAVE_STRUCT_IN6_ADDR
    struct in6_addr ip6;
#endif
};

/*
 * IP address and netmask pairs for checking against local interfaces.
 */
struct interface {
    SLIST_ENTRY(interface) entries;
    unsigned int family;	/* AF_INET or AF_INET6 */
    union sudo_in_addr_un addr;
    union sudo_in_addr_un netmask;
};

SLIST_HEAD(interface_list, interface);

/*
 * Prototypes for external functions.
 */
int get_net_ifs(char **addrinfo);
void dump_interfaces(const char *);
bool set_interfaces(const char *);
struct interface_list *get_interfaces(void);

#endif /* SUDOERS_INTERFACES_H */
