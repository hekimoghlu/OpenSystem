/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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
#include "packer.h"

int
main(argc, argv)
    int argc;
    char *argv[];
{
    int32 i;
    PWDICT *pwp;
    char buffer[STRINGSIZE];

    if (argc <= 1)
    {
	fprintf(stderr, "Usage:\t%s dbname\n", argv[0]);
	return (-1);
    }

    if (!(pwp = PWOpen (argv[1], "r")))
    {
	perror ("PWOpen");
	return (-1);
    }

    for (i=0; i < PW_WORDS(pwp); i++)
    {
    	register char *c;

	if (!(c = (char *) GetPW (pwp, i)))
	{
	    fprintf(stderr, "error: GetPW %d failed\n", i);
	    continue;
	}

	printf ("%s\n", c);
    }

    return (0);
}
