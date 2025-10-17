/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#include <string.h>
#include <ctype.h>
#include <openssl/e_os2.h>
#include <openssl/buffer.h>
#include <openssl/crypto.h>

int main(int argc, char *argv[])
	{
	char *p, *q = 0, *program;

	p = strrchr(argv[0], '/');
	if (!p) p = strrchr(argv[0], '\\');
#ifdef OPENSSL_SYS_VMS
	if (!p) p = strrchr(argv[0], ']');
	if (p) q = strrchr(p, '>');
	if (q) p = q;
	if (!p) p = strrchr(argv[0], ':');
	q = 0;
#endif
	if (p) p++;
	if (!p) p = argv[0];
	if (p) q = strchr(p, '.');
	if (p && !q) q = p + strlen(p);

	if (!p)
		program = BUF_strdup("(unknown)");
	else
		{
		program = OPENSSL_malloc((q - p) + 1);
		strncpy(program, p, q - p);
		program[q - p] = '\0';
		}

	for(p = program; *p; p++)
		if (islower((unsigned char)(*p)))
			*p = toupper((unsigned char)(*p));

	q = strstr(program, "TEST");
	if (q > p && q[-1] == '_') q--;
	*q = '\0';

	printf("No %s support\n", program);

	OPENSSL_free(program);
	return(0);
	}
