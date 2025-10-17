/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pwd.h>
#include <netinet/in.h>
#include <rpc/types.h>
#include <rpc/xdr.h>
#include <rpc/rpc.h>
#include <rpcsvc/yp_prot.h>
#include <rpcsvc/ypclnt.h>
#include <rpcsvc/yppasswd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/file.h>
#include <errno.h>

#include <OpenDirectory/OpenDirectory.h>

#include "passwd.h"

//-------------------------------------------------------------------------------------
//	od_check_passwd
//-------------------------------------------------------------------------------------

int
od_check_passwd(const char *uname, const char *domain)
{
	int	authenticated = 0;

	ODSessionRef	session = NULL;
	ODNodeRef		node = NULL;
	ODRecordRef		rec = NULL;
	CFStringRef		user = NULL;
	CFStringRef		location = NULL;
	CFStringRef		password = NULL;

	if (uname) user = CFStringCreateWithCString(NULL, uname, kCFStringEncodingUTF8);
	if (domain) location = CFStringCreateWithCString(NULL, domain, kCFStringEncodingUTF8);

	if (user) {
		printf("Checking password for %s.\n", uname);
		char* p = getpass("Password:");
		if (p) password = CFStringCreateWithCString(NULL, p, kCFStringEncodingUTF8);
	}

	if (password) {
		session = ODSessionCreate(NULL, NULL, NULL);
		if (session) {
			if (location) {
				node = ODNodeCreateWithName(NULL, session, location, NULL);
			} else {
				node = ODNodeCreateWithNodeType(NULL, session, kODNodeTypeAuthentication, NULL);
			}
			if (node) {
				rec = ODNodeCopyRecord(node, kODRecordTypeUsers, user, NULL, NULL);
			}
			if (rec) {
				authenticated = ODRecordVerifyPassword(rec, password, NULL);
			}
		}
	}

	if (!authenticated) {
		fprintf(stderr, "Sorry\n");
		exit(1);
	}

	return 0;
}
