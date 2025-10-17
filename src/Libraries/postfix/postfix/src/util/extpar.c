/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 12, 2023.
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
/*
  * System library.
  */
#include <sys_defs.h>
#include <ctype.h>

 /*
  * Utility library.
  */
#include <msg.h>
#include <stringops.h>

/* extpar - extract text from parentheses */

char   *extpar(char **bp, const char *parens, int flags)
{
    char   *cp = *bp;
    char   *err = 0;
    size_t  len;

    if (cp[0] != parens[0])
	msg_panic("extpar: no '%c' at start of text: \"%s\"", parens[0], cp);
    if ((len = balpar(cp, parens)) == 0) {
	err = concatenate("missing '", parens + 1, "' in \"",
			  cp, "\"", (char *) 0);
	cp += 1;
    } else {
	if (cp[len] != 0)
	    err = concatenate("syntax error after '", parens + 1, "' in \"",
			      cp, "\"", (char *) 0);
	cp += 1;
	cp[len -= 2] = 0;
    }
    if (flags & EXTPAR_FLAG_STRIP) {
	trimblanks(cp, len)[0] = 0;
	while (ISSPACE(*cp))
	    cp++;
    }
    *bp = cp;
    return (err);
}
