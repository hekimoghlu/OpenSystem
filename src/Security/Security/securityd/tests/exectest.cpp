/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
//
// Exectest - privileged-execution test driver
//
#include <Security/Authorization.h>
#include <unistd.h>
#include <stdlib.h>


void doLoopback(int argc, char *argv[]);


int main(int argc, char **argv)
{
	const char *path = "/usr/bin/id";
	bool writeToPipe = false;
	bool loopback = false;
	
	int arg;
	extern char *optarg;
	extern int optind;
	while ((arg = getopt(argc, argv, "f:lLw")) != -1) {
		switch (arg) {
		case 'f':
			path = optarg;
			break;
		case 'l':
			loopback = true;
			break;
		case 'L':
			doLoopback(argc, argv);
			exit(0);
		case 'w':
			writeToPipe = true;
			break;
		case '?':
			exit(2);
		}
	}
	
	AuthorizationItem right = { "system.privilege.admin", 0, NULL, 0 };
	AuthorizationRights rights = { 1, &right };

	AuthorizationRef auth;
	if (OSStatus error = AuthorizationCreate(&rights, NULL /*env*/,
		kAuthorizationFlagInteractionAllowed |
		kAuthorizationFlagExtendRights |
		kAuthorizationFlagPreAuthorize,
		&auth)) {
		printf("create error %ld\n", error);
		exit(1);
	}
	
	if (loopback) {
		path = argv[0];
		argv[--optind] = "-L";	// backing over existing array element
	}
	
	FILE *f;
	if (OSStatus error = AuthorizationExecuteWithPrivileges(auth,
		path, 0, argv + optind, &f)) {
		printf("exec error %ld\n", error);
		exit(1);
	}
	printf("--- execute successful ---\n");
	if (writeToPipe) {
		char buffer[1024];
		while (fgets(buffer, sizeof(buffer), stdin))
			fprintf(f, "%s", buffer);
	} else {
		char buffer[1024];
		while (fgets(buffer, sizeof(buffer), f))
			printf("%s", buffer);
	}
	printf("--- end of output ---\n");
	exit(0);
}


void doLoopback(int argc, char *argv[])
{
	// general status
	printf("Authorization Execution Loopback Test\n");
	printf("Invoked as");
	for (int n = 0; argv[n]; n++)
		printf(" %s", argv[n]);
	printf("\n");
	
	// recover the authorization handle
	AuthorizationRef auth;
	if (OSStatus err = AuthorizationCopyPrivilegedReference(&auth, 0)) {
		printf("Cannot recover AuthorizationRef: error=%ld\n", err);
		exit(1);
	}
	
	printf("AuthorizationRef recovered.\n");
}
