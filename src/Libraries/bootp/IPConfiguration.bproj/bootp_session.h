/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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
 * bootp_session.h
 * - maintain BOOTP client socket session
 * - maintain list of BOOTP clients
 * - distribute packet reception to enabled clients
 */

/* 
 * Modification History
 *
 * May 10, 2000		Dieter Siegmund (dieter@apple.com)
 * - created
 */

#ifndef _S_BOOTP_SESSION_H
#define _S_BOOTP_SESSION_H

#include <stdint.h>
#include "dhcp_options.h"
#include "FDSet.h"
#include "interfaces.h"

typedef struct {
    struct dhcp  *		data;
    int				size;
    dhcpol_t			options;
} bootp_receive_data_t;

/*
 * Type: bootp_receive_func_t
 * Purpose:
 *   Called to deliver data to the client.  The first two args are
 *   supplied by the client, the third is a pointer to a bootp_receive_data_t.
 */
typedef void (bootp_receive_func_t)(void * arg1, void * arg2, void * arg3);

typedef struct bootp_client * bootp_client_t;

bootp_client_t
bootp_client_init(interface_t * if_p);

void
bootp_client_free(bootp_client_t * session);

void
bootp_client_enable_receive(bootp_client_t client,
			    bootp_receive_func_t * func, 
			    void * arg1, void * arg2);

void
bootp_client_disable_receive(bootp_client_t client);

int
bootp_client_transmit(bootp_client_t client,
		      struct in_addr dest_ip,
		      struct in_addr src_ip,
		      uint16_t dest_port,
		      uint16_t src_port,
		      void * data, int len);

void
bootp_session_init(uint16_t client_port);

void
bootp_session_set_verbose(bool verbose);

#endif /* _S_BOOTP_SESSION_H */
