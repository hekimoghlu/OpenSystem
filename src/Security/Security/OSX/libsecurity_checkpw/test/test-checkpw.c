/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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



#include <security/checkpw.h>
#include <pwd.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

const char *prompt = "checkpw test prompt:";

int
main(int argv, char *argc[])
{
	char *uname;
	int retval = 0;
	struct passwd *pw = NULL;

	uname = (char*)getenv("USER");
	if ( NULL == uname)
	{
		uid_t uid = getuid();
		struct passwd *pw = getpwuid(uid);
		uname = pw->pw_name;
	}

	retval = checkpw(uname, getpass(prompt));
	if (0 == retval)
	{
		printf("Password is okay.\n");
	} else {
		printf("Incorrect password.\n");
	}

	return retval;
}
