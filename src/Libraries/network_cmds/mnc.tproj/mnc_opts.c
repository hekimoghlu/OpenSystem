/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef WINDOWS

/* UNIX-y includes */
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#else

/* WINDOWS-y includes */
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "mnc.h"

/* Display a usage statement */
void usage(void)
{
	fprintf(stderr, 
		"Usage: mnc [-l] [-i interface] [-p port] group-id "
		"[source-address]\n\n"
		"-l :    listen mode\n"
		"-i :    specify interface to listen\n"
		"-p :    specify port to listen/send on\n\n");
	exit(1);
}

struct mnc_configuration * parse_arguments(int argc, char **argv)
{
	/* Utility variables */
	int					optind,
						errorcode;
	struct	addrinfo			hints;

	/* Our persisting configuration */
	static	struct mnc_configuration	config;

	/* Set some defaults */
	config.mode	= SENDER;
	config.port 	= MNC_DEFAULT_PORT;
	config.iface	= NULL;
	config.source	= NULL;

	/* Loop through the arguments */
	for (optind = 1; optind < (argc - 1); optind++)
	{
		if ( (argv[optind][0] == '-') || (argv[optind][0] == '/') )
		{
			switch(argv[optind][1])
			{
				/* Set listening mode */
				case 'l':	config.mode = LISTENER;
						break;

				/* Set port */
				case 'p':	config.port = argv[++optind];
						break;

				/* Set an interface */
				case 'i':	config.iface = argv[++optind];
						break;

				/* Unrecognised option */
				default:	usage();
						break;
			}
		}
		else
		{
			/* assume we've ran out of options */
			break;
		}
	}

	/* There's a chance we were passed one option */
	if (optind >= argc || argv[optind][0] == '-')
	{
		usage();
	}

	/* Now make sure we have either exactly 1 or 2 more arguments */
	if ( (argc - optind) != 1 && (argc - optind) != 2 )
	{
		/* We have not been given the right ammount of 
		   arguments */
		usage();
	}

	/* You can't have an interface without also listening */
	if (config.mode == SENDER && config.iface != NULL)
	{
		mnc_error("You may only specify the interface when in"
				" listening mode\n");
	}

	/* Set some hints for getaddrinfo */
	memset(&hints, 0, sizeof(hints));
	
	/* We want a UDP socket */
	hints.ai_socktype = SOCK_DGRAM;

	/* Don't do any name-lookups */
	hints.ai_flags = AI_NUMERICHOST;
	
	/* Get the group-id information */
	if ( (errorcode =
	      getaddrinfo(argv[optind], config.port, &hints, &config.group)) != 0)
	{
		mnc_error("Error getting group-id address information: %s\n", 
			  gai_strerror(errorcode));
	}

	/* Move on to next argument */
	optind++;
	
	/* Get the source information */
	if ( (argc - optind) == 1)
	{

		if ( (errorcode = 
        	      getaddrinfo(argv[optind], config.port, &hints, &config.source)) 
		    != 0)
		{
			mnc_error("Error getting source-address information: %s\n", 
			          gai_strerror(errorcode));	
		}
	
		/* Confirm that the source and group are in the same Address Family */
		if ( config.source->ai_family != config.group->ai_family )
		{
			mnc_error("Group ID and Source address are not of "
				  "the same type\n");
		}
	}

	return &config;
}
