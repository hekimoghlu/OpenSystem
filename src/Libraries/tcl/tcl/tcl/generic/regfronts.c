/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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
#include "regguts.h"

/*
 - regcomp - compile regular expression
 */
int
regcomp(
    regex_t *re,
    CONST char *str,
    int flags)
{
    size_t len;
    int f = flags;

    if (f&REG_PEND) {
	len = re->re_endp - str;
	f &= ~REG_PEND;
    } else {
	len = strlen(str);
    }

    return re_comp(re, str, len, f);
}

/*
 - regexec - execute regular expression
 */
int
regexec(
    regex_t *re,
    CONST char *str,
    size_t nmatch,
    regmatch_t pmatch[],
    int flags)
{
    CONST char *start;
    size_t len;
    int f = flags;

    if (f & REG_STARTEND) {
	start = str + pmatch[0].rm_so;
	len = pmatch[0].rm_eo - pmatch[0].rm_so;
	f &= ~REG_STARTEND;
    } else {
	start = str;
	len = strlen(str);
    }

    return re_exec(re, start, len, nmatch, pmatch, f);
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
