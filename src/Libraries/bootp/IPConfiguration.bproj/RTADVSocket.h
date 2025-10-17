/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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
 * RTADVSocket.h
 * - maintain list of Router Advertisement client "sockets"
 * - distribute packet reception to enabled "sockets"
 */

/* 
 * Modification History
 *
 * June 4, 2010		Dieter Siegmund (dieter@apple.com)
 * - created (based on DHCPv6Socket.h)
 */

#ifndef _S_RTADVSOCKET_H
#define _S_RTADVSOCKET_H

#include <stdint.h>
#include <stdbool.h>
#include "FDSet.h"
#include "interfaces.h"
#include "RouterAdvertisement.h"

/*
 * Type: RTADVSocketReceiveFunc
 * Purpose:
 *   Called to deliver data to the client.  The first two args are
 *   supplied by the client, the third is a RouterAdvertisementRef.
 */
typedef void (RTADVSocketReceiveFunc)(void * arg1, void * arg2, void * arg3);
typedef RTADVSocketReceiveFunc * RTADVSocketReceiveFuncPtr;

typedef struct RTADVSocket * RTADVSocketRef;

RTADVSocketRef
RTADVSocketCreate(interface_t * if_p);

interface_t *
RTADVSocketGetInterface(RTADVSocketRef sock);

void
RTADVSocketRelease(RTADVSocketRef * sock);

void
RTADVSocketEnableReceive(RTADVSocketRef sock,
			 RTADVSocketReceiveFuncPtr func, 
			 void * arg1, void * arg2);

void
RTADVSocketDisableReceive(RTADVSocketRef sock);

errno_t
RTADVSocketSendSolicitation(RTADVSocketRef sock, bool lladdr_ok);

#endif /* _S_RTADVSOCKET_H */
