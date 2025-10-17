/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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
#include <string.h>

#ifndef NO_EAI
#include <unicode/uidna.h>

 /*
  * Utility library.
  */
#include <vstring.h>
#include <stringops.h>
#include <midna_domain.h>

 /*
  * Global library.
  */
#include <midna_adomain.h>

#define STR(x)	vstring_str(x)

/* midna_adomain_to_utf8 - convert address domain portion to UTF8 */

char   *midna_adomain_to_utf8(VSTRING *dest, const char *src)
{
    const char *cp;
    const char *domain_utf8;

    if ((cp = strrchr(src, '@')) == 0) {
	vstring_strcpy(dest, src);
    } else {
	vstring_sprintf(dest, "%*s@", (int) (cp - src), src);
	if (*(cp += 1)) {
	    if (allascii(cp) && strstr(cp, "--") == 0) {
		vstring_strcat(dest, cp);
	    } else if ((domain_utf8 = midna_domain_to_utf8(cp)) == 0) {
		return (0);
	    } else {
		vstring_strcat(dest, domain_utf8);
	    }
	}
    }
    return (STR(dest));
}

/* midna_adomain_to_ascii - convert address domain portion to ASCII */

char   *midna_adomain_to_ascii(VSTRING *dest, const char *src)
{
    const char *cp;
    const char *domain_ascii;

    if ((cp = strrchr(src, '@')) == 0) {
	vstring_strcpy(dest, src);
    } else {
	vstring_sprintf(dest, "%*s@", (int) (cp - src), src);
	if (*(cp += 1)) {
	    if (allascii(cp)) {
		vstring_strcat(dest, cp);
	    } else if ((domain_ascii = midna_domain_to_ascii(cp + 1)) == 0) {
		return (0);
	    } else {
		vstring_strcat(dest, domain_ascii);
	    }
	}
    }
    return (STR(dest));
}

#endif					/* NO_IDNA */
