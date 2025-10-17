/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 21, 2022.
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
#ifndef __PCAP_RPCAP_INT_H__
#define __PCAP_RPCAP_INT_H__

#include "pcap.h"
#include "sockutils.h"	/* Needed for some structures (like SOCKET, sockaddr_in) which are used here */

/*
 * \file pcap-rpcap-int.h
 *
 * This file keeps all the definitions used by the RPCAP client and server,
 * other than the protocol definitions.
 *
 * \warning All the RPCAP functions that are allowed to return a buffer containing
 * the error description can return max PCAP_ERRBUF_SIZE characters.
 * However there is no guarantees that the string will be zero-terminated.
 * Best practice is to define the errbuf variable as a char of size 'PCAP_ERRBUF_SIZE+1'
 * and to insert manually the termination char at the end of the buffer. This will
 * guarantee that no buffer overflows occur even if we use the printf() to show
 * the error on the screen.
 */

/*********************************************************
 *                                                       *
 * General definitions / typedefs for the RPCAP protocol *
 *                                                       *
 *********************************************************/

/*
 * \brief Buffer used by socket functions to send-receive packets.
 * In case you plan to have messages larger than this value, you have to increase it.
 */
#define RPCAP_NETBUF_SIZE 64000

/*********************************************************
 *                                                       *
 * Exported function prototypes                          *
 *                                                       *
 *********************************************************/
void rpcap_createhdr(struct rpcap_header *header, uint8 type, uint16 value, uint32 length);
int rpcap_senderror(SOCKET sock, char *error, unsigned short errcode, char *errbuf);

#endif
