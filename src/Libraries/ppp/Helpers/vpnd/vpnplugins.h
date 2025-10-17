/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#ifndef __VPNPLUGINS_H__
#define __VPNPLUGINS_H__

#include "vpnoptions.h"

/*
 * This struct contains pointers to a set of procedures for
 * doing operations on a "vpn_channel".  A vpn_channel provides
 * functionality for vpnd to listen for and accept connections
 * for a paticular VPN protocol.  After a connection is accepted
 * the vpn plugin will fork and exec a copy of pppd to handle the
 * connection.
 *
 *		return values for refuse:	
 *			-1 			error 
 *			socket# 	launch pppd and notify caller that server is full
 *			0			handled - do not launch pppd
 */

struct vpn_channel {
    /* read and allocate args to pass to pppd */
    int (*get_pppd_args) __P((struct vpn_params*, int));
    /* intialize the vpn plugin */
    int (*listen) __P((void));
    /* accept an incoming connection */
    int (*accept) __P((void));
    /* refuse an incoming connection */
    int (*refuse) __P((void));
    /* we're finished with the channel */
    void (*close) __P((void));
    /* health check function */
    int (*health_check) __P((int *, int));
    /* load balance redirect function */
    int (*lb_redirect) __P((struct in_addr *, struct in_addr *));
};
   
void init_address_lists(void);
int add_address(char* ip_address);
int add_address_range(char* ip_addr_start, char* ip_addr_end);
void begin_address_update(void);
void cancel_address_update(void);
void apply_address_update(void);
int address_avail(void);
int init_plugin(struct vpn_params *params);
int get_plugin_args(struct vpn_params* params, int reload);
void accept_connections(struct vpn_params* params);

#endif
