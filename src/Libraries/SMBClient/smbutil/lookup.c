/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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
#include <sys/param.h>
#include <sys/errno.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <err.h>
#include <stdio.h>
#include <unistd.h>
#include <strings.h>
#include <stdlib.h>
#include <sysexits.h>

#include <smbclient/smbclient.h>
#include <smbclient/netbios.h>
#include <arpa/inet.h>

#include "common.h"

/*
 * Make a copy of the name and if request unpercent escape the name
 */
static char * 
getHostName(const char *name, Boolean escapeNames)
{	
	char *newName = strdup(name);
	CFStringRef nameRef = NULL;
	CFStringRef newNameRef = NULL;
		
	/* They don't want it escape or the strdup failed */
	if (!escapeNames || !newName) {
		return newName;
	}
	/* Get a CFString */
	nameRef = CFStringCreateWithCString(kCFAllocatorDefault, name, kCFStringEncodingUTF8);
	
	/* unpercent escape out the CFString, but leave Space escaped (?) */
	if (nameRef) {
		newNameRef = CFURLCreateStringByReplacingPercentEscapes(kCFAllocatorDefault,
																nameRef, CFSTR(" "));
	}
	/* Now create an unpercent escape out c style string */
	if (newNameRef) {
		int maxlen = (int)CFStringGetLength(newNameRef)+1;
		char *tempName = malloc(maxlen);
		
		if (tempName) {
			free(newName);
			newName = tempName;
			CFStringGetCString(newNameRef, newName, maxlen, kCFStringEncodingUTF8);
		}	
	}
	
	if (nameRef) {
		CFRelease(nameRef);
	}
	if (newNameRef) {
		CFRelease(newNameRef);
	}
	return newName;
}


int
cmd_lookup(int argc, char *argv[])
{
	char *hostname;
	int opt;
	struct sockaddr_storage *startAddr = NULL, *listAddr = NULL;
	const char *winsServer = NULL;
	int32_t ii, count = 0;
	struct sockaddr_storage respAddr;
	struct sockaddr_in *in4 = NULL;
	struct sockaddr_in6 *in6 = NULL;
	char addrStr[INET6_ADDRSTRLEN+1];
	uint8_t nodeType = kNetBIOSFileServerService;
	Boolean escapeNames= FALSE;
	
	bzero(&respAddr, sizeof(respAddr));
	if (argc < 2)
		lookup_usage();
	while ((opt = getopt(argc, argv, "ew:t:")) != EOF) {
		switch(opt) {
			case 'e':
				escapeNames = TRUE;
				break;
			case 'w':
				winsServer = optarg;
				break;
			case 't':
				errno = 0;
				nodeType = (uint8_t)strtol(optarg, NULL, 0);
				if (errno)
					errx(EX_DATAERR, "invalid value for node type");
				break;
		    default:
				lookup_usage();
				/*NOTREACHED*/
		}
	}
	if (optind >= argc)
		lookup_usage();

	hostname = getHostName(argv[argc - 1], escapeNames);
	if (!hostname) {
		err(EX_OSERR, "failed to resolve %s", argv[argc - 1]);
	}
	
	startAddr = listAddr = SMBResolveNetBIOSNameEx(hostname, nodeType, winsServer, 
													0, &respAddr, &count);
	if (startAddr == NULL) {
		err(EX_NOHOST, "unable to resolve %s", hostname);
	}
	
	if (respAddr.ss_family == AF_INET) {
		in4 = (struct sockaddr_in *)&respAddr;
		inet_ntop(respAddr.ss_family, &in4->sin_addr, addrStr, sizeof(addrStr));
	} else if (respAddr.ss_family == AF_INET6) {
		in6 = (struct sockaddr_in6 *)&respAddr;
		inet_ntop(respAddr.ss_family, &in6->sin6_addr, addrStr, sizeof(addrStr));
	} else {
		strcpy(addrStr, "unknown address family");
	}

	fprintf(stdout, "Got response from %s\n", addrStr);

	for (ii=0; ii < count; ii++) {
		if (listAddr->ss_family == AF_INET) {
			in4 = (struct sockaddr_in *)listAddr;
			inet_ntop(listAddr->ss_family, &in4->sin_addr, addrStr, sizeof(addrStr));
		} else if (respAddr.ss_family == AF_INET6) {
			in6 = (struct sockaddr_in6 *)listAddr;
			inet_ntop(respAddr.ss_family, &in6->sin6_addr, addrStr, sizeof(addrStr));
		} else {
			strcpy(addrStr, "unknown address family");
		}
		fprintf(stdout, "IP address of %s: %s\n", hostname, addrStr);
		listAddr++;
	}
	if (startAddr) {
		free(startAddr);
	}
	if (hostname) {
		free(hostname);
	}
	return 0;
}


void
lookup_usage(void)
{
	fprintf(stderr, "usage: smbutil lookup [-e] [-w host] [-t node type] name\n");
	exit(1);
}
