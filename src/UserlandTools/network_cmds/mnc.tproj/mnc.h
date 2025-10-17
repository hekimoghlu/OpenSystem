/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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
#ifndef _MNC_H_
#define _MNC_H_

#ifndef WINDOWS

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#else

#include <winsock2.h>
#include <ws2tcpip.h>

#endif

/* The UDP port MNC will use by default */
#define MNC_DEFAULT_PORT    	"1234"

struct mnc_configuration
{
	/* Are we sending or recieving ? */
	enum {SENDER, LISTENER}	mode;

	/* What UDP port are we using ? */
	char	*		port;
	
	/* The group-id */
	struct addrinfo	*	group;

	/* The source */
	struct addrinfo *	source;
	
	/* An interface index for listening */
	char	*		iface;
};


/* Functions in mnc_opts.c */
void 				usage(void);
struct mnc_configuration * 	parse_arguments(int argc, char **argv);

/* Functions in mnc_multicast.c */
int multicast_setup_listen(int, struct addrinfo *, struct addrinfo *, char *);
int multicast_setup_send(int, struct addrinfo *, struct addrinfo *);

#include <err.h>

#define mnc_warning(fmt, ...) warnx(fmt, ##__VA_ARGS__)
#define mnc_error(fmt, ...) errx(1, fmt, ##__VA_ARGS__)

#endif /* _MNC_H_ */
