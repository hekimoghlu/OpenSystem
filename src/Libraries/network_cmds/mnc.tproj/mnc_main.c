/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#ifndef WINDOWS

/* Non-windows includes */

#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdio.h>

#else 

/* Windows-specific includes */

#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>

#endif /* WINDOWS */

#include "mnc.h"

int main(int argc, char **argv)
{
	/* Utility variables */
	int				sock,
					len;
	char				buffer[1024];

	/* Our main configuration */
	struct mnc_configuration *	config;

#ifdef WINDOWS
	WSADATA 			wsaData;
 
	if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0)
	{
		mnc_error("This operating system is not supported\n");
	}
#endif
	
	/* Parse the command line */
	config = parse_arguments(argc, argv);
	
	/* Create a socket */
	if ((sock = socket(config->group->ai_family, config->group->ai_socktype, 
 	    config->group->ai_protocol)) < 0)
	{
		mnc_error("Could not create socket\n");
	}

	/* Are we supposed to listen? */
	if (config->mode == LISTENER)
	{
		/* Set up the socket for listening */
		if (multicast_setup_listen(sock, config->group, config->source, 
		                 config->iface) < 0)
		{
			mnc_error("Can not listen for multicast packets.\n");
		}

		/* Recieve the packets */
		while ((len = recvfrom(sock, buffer, sizeof(buffer), 
		                       0, NULL, NULL)) >= 0)
		{	
			write(STDOUT_FILENO, buffer, len);
		}
	}
	else /* Assume MODE == SENDER */
	{
		/* Set up the socket for sending */
		if (multicast_setup_send(sock, config->group, config->source) 
		    < 0)
		{
			mnc_error("Can not send multicast packets\n");
		}
		
		/* Send the packets */
		while((len = read(STDIN_FILENO, buffer, sizeof(buffer))) > 0)
		{
			sendto(sock, buffer, len, 0, config->group->ai_addr, 
			       config->group->ai_addrlen);
		}
	}
	
	/* Close the socket */
	close(sock);

	return 0;
}
