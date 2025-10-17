/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 1, 2024.
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
/* Copyright (C) 1990-2, RSA Data Security, Inc. Created 1990. All rights
 * reserved.
 * 
 * RSA Data Security, Inc. makes no representations concerning either the
 * merchantability of this software or the suitability of this software for
 * any particular purpose. It is provided "as is" without express or implied
 * warranty of any kind.
 * 
 * These notices must be retained in any copies of any part of this
 * documentation and/or software. */

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>

#include <stdio.h>
#include <time.h>
#include <string.h>

/* The following makes MD default to MD5 if it has not already been defined
 * with C compiler flags. */
#ifndef MD
#define MD 5
#endif

#if MD == 2
#include "md2.h"
#define MDData MD2Data
#endif
#if MD == 4
#include "md4.h"
#define MDData MD4Data
#endif
#if MD == 5
#include "md5.h"
#define MDData MD5Data
#endif

/* Digests a string and prints the result. */
static void 
MDString(char *string)
{
	char buf[33];

	printf("MD%d (\"%s\") = %s\n",
	       MD, string, MDData(string, strlen(string), buf));
}

/* Digests a reference suite of strings and prints the results. */
int
main(void)
{
	printf("MD%d test suite:\n", MD);

	MDString("");
	MDString("a");
	MDString("abc");
	MDString("message digest");
	MDString("abcdefghijklmnopqrstuvwxyz");
	MDString("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"abcdefghijklmnopqrstuvwxyz0123456789");
	MDString("1234567890123456789012345678901234567890"
		"1234567890123456789012345678901234567890");

	return 0;
}
