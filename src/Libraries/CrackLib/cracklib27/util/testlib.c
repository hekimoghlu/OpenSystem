/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
main ()
{
    int32 i;
    char buffer[STRINGSIZE];

    printf("enter potential passwords, one per line...\n");

    while (fgets (buffer, STRINGSIZE, stdin))
    {
    	char *val;
 
	Chop(buffer);

	val = FascistCheck(buffer, CRACKLIB_DICTPATH);

	if (!val)
	{
		val = "ok";
	}	
	printf ("%s: %s\n", buffer, val);
    }

    return (0);
}
