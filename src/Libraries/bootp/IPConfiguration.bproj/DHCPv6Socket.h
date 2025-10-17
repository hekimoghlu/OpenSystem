/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 8, 2024.
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
 * DHCPv6Socket.h
 * - maintain list of DHCPv6 clients
 * - distribute packet reception to enabled clients
 */
/* 
 * Modification History
 *
 * September 30, 2009		Dieter Siegmund (dieter@apple.com)
 * - created (based on bootp_session.h)
 */

#ifndef _S_DHCPV6SOCKET_H
#define _S_DHCPV6SOCKET_H

#include <stdint.h>
#include "DHCPv6.h"
#include "DHCPv6Options.h"
#include "FDSet.h"
#include "interfaces.h"

typedef struct {
    DHCPv6PacketRef		pkt;
    int				pkt_len;
    DHCPv6OptionListRef		options;
} DHCPv6SocketReceiveData, * DHCPv6SocketReceiveDataRef;

/*
 * Type: DHCPv6SocketReceiveFunc
 * Purpose:
 *   Called to deliver data to the client.  The first two args are
 *   supplied by the client, the third is a DHCPv6ReceiveDataRef.
 */
typedef void (DHCPv6SocketReceiveFunc)(void * arg1, void * arg2, void * arg3);
typedef DHCPv6SocketReceiveFunc * DHCPv6SocketReceiveFuncPtr;

typedef struct DHCPv6Socket * DHCPv6SocketRef;

void
DHCPv6SocketSetVerbose(bool verbose);

void
DHCPv6SocketSetPorts(uint16_t client_port, uint16_t server_port);

DHCPv6SocketRef
DHCPv6SocketCreate(interface_t * if_p);

interface_t *
DHCPv6SocketGetInterface(DHCPv6SocketRef sock);

void
DHCPv6SocketRelease(DHCPv6SocketRef * sock);

void
DHCPv6SocketEnableReceive(DHCPv6SocketRef sock,
			  DHCPv6TransactionID transaction_id,
			  DHCPv6SocketReceiveFuncPtr func, 
			  void * arg1, void * arg2);

bool
DHCPv6SocketReceiveIsEnabled(DHCPv6SocketRef sock);

void
DHCPv6SocketDisableReceive(DHCPv6SocketRef sock);

errno_t
DHCPv6SocketTransmit(DHCPv6SocketRef sock,
		     DHCPv6PacketRef pkt, int pkt_len);

#endif /* _S_DHCPV6SOCKET_H */
