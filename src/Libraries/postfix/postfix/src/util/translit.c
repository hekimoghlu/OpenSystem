/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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
/* System library. */

#include "sys_defs.h"
#include <string.h>

/* Utility library. */

#include "stringops.h"

char   *translit(char *string, const char *original, const char *replacement)
{
    char   *cp;
    const char *op;

    /*
     * For large inputs, should use a lookup table.
     */
    for (cp = string; *cp != 0; cp++) {
	for (op = original; *op != 0; op++) {
	    if (*cp == *op) {
		*cp = replacement[op - original];
		break;
	    }
	}
    }
    return (string);
}

#ifdef TEST

 /*
  * Usage: translit string1 string2
  * 
  * test program to perform the most basic operation of the UNIX tr command.
  */
#include <msg.h>
#include <vstring.h>
#include <vstream.h>
#include <vstring_vstream.h>

#define STR	vstring_str

int     main(int argc, char **argv)
{
    VSTRING *buf = vstring_alloc(100);

    if (argc != 3)
	msg_fatal("usage: %s string1 string2", argv[0]);
    while (vstring_fgets(buf, VSTREAM_IN))
	vstream_fputs(translit(STR(buf), argv[1], argv[2]), VSTREAM_OUT);
    vstream_fflush(VSTREAM_OUT);
    vstring_free(buf);
    return (0);
}

#endif
