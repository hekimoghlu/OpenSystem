/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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
#include <config.h>

RCSID("$Id$");

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <roken.h>
#ifdef SOCKS
#include <socks.h>
#endif
#include "misc.h"
#include "auth.h"
#include "encrypt.h"


const char *RemoteHostName;
const char *LocalHostName;
char *UserNameRequested = 0;
int ConnectedCount = 0;

void
auth_encrypt_init(const char *local, const char *remote, const char *name,
		  int server)
{
    RemoteHostName = remote;
    LocalHostName = local;
#ifdef AUTHENTICATION
    auth_init(name, server);
#endif
#ifdef ENCRYPTION
    encrypt_init(name, server);
#endif
    if (UserNameRequested) {
	free(UserNameRequested);
	UserNameRequested = 0;
    }
}

void
auth_encrypt_user(const char *name)
{
    if (UserNameRequested)
	free(UserNameRequested);
    UserNameRequested = name ? strdup(name) : 0;
}

void
auth_encrypt_connect(int cnt)
{
}

void
printd(const unsigned char *data, int cnt)
{
    if (cnt > 16)
	cnt = 16;
    while (cnt-- > 0) {
	printf(" %02x", *data);
	++data;
    }
}
