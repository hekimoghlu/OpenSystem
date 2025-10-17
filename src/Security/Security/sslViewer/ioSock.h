/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
#ifndef	_IO_SOCK_H_
#define _IO_SOCK_H_

#include <Security/SecureTransport.h>
#include <sys/types.h>

#ifdef	__cplusplus
extern "C" {
#endif

/*
 * Opaque reference to an Open Transport connection.
 */
typedef int otSocket;

/*
 * info about a peer returned from MakeServerConnection() and 
 * AcceptClientConnection().
 */
typedef struct
{   UInt32      ipAddr;
    int         port;
} PeerSpec;

/*
 * Ont-time only init.
 */
void initSslOt(void);

/*
 * Connect to server. 
 */
extern OSStatus MakeServerConnection(
	const char *hostName, 
	int port, 
	int nonBlocking,		// 0 or 1
	otSocket *socketNo, 	// RETURNED
	PeerSpec *peer);		// RETURNED

/*
 * Set up an otSocket to listen for client connections. Call once, then
 * use multiple AcceptClientConnection calls. 
 */
OSStatus ListenForClients(
	int port, 
	int nonBlocking,		// 0 or 1
	otSocket *socketNo); 	// RETURNED

/*
 * Accept a client connection. Call endpointShutdown() for each successful;
 * return from this function. 
 */
OSStatus AcceptClientConnection(
	otSocket listenSock, 		// obtained from ListenForClients
	otSocket *acceptSock, 		// RETURNED
	PeerSpec *peer);			// RETURNED

/*
 * Shut down a connection.
 */
void endpointShutdown(
	otSocket socket);
	
/*
 * R/W. Called out from SSL.
 */
OSStatus SocketRead(
	SSLConnectionRef 	connection,
	void 				*data, 			/* owned by 
	 									 * caller, data
	 									 * RETURNED */
	size_t 				*dataLength);	/* IN/OUT */ 
	
OSStatus SocketWrite(
	SSLConnectionRef 	connection,
	const void	 		*data, 
	size_t 				*dataLength);	/* IN/OUT */ 

#ifdef	__cplusplus
}
#endif

#endif	/* _IO_SOCK_H_ */
