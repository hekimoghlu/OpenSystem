/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>

#include <err.h>
#include <iconv.h>
#include <stdlib.h>
#include <string.h>

int
main(void)
{
	iconv_t cd;
	size_t inbytes, outbytes;
	char *str1 = "FOOBAR";
	const char *str2 = "FOOBAR";
	char ** in1;
	const char ** in2 = &str2;
	char *out1, *out2;

	inbytes = outbytes = strlen("FOOBAR");

	if ((cd = iconv_open("UTF-8", "ASCII")) == (iconv_t)-1)
		err(1, NULL);

	if ((out2 = malloc(inbytes)) == NULL)
		err(1, NULL);

	if (iconv(cd, in2, &inbytes, &out2, &outbytes) == -1)
		err(1, NULL);

	in1 = &str1;
	inbytes = outbytes = strlen("FOOBAR");

	if ((out1 = malloc(inbytes)) == NULL)
		err(1, NULL);

	if (iconv(cd, in1, &inbytes, &out1, &outbytes) == -1)
		err(1, NULL);

	return (EXIT_SUCCESS);

}
