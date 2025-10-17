/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 14, 2025.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_util.h"
#include "sudo_eventlog.h"

size_t
eventlog_writeln(FILE *fp, char *line, size_t linelen, size_t maxlen)
{
    const char *indent = "";
    char *beg = line;
    char *end;
    int len;
    size_t outlen = 0;
    debug_decl(eventlog_writeln, SUDO_DEBUG_UTIL);

    if (maxlen < sizeof(EVENTLOG_INDENT)) {
	/* Maximum length too small, disable wrapping. */
	outlen = fwrite(line, 1, linelen, fp);
	if (outlen != linelen)
	    debug_return_ssize_t(-1);
	if (fputc('\n', fp) == EOF)
	    debug_return_ssize_t(-1);
	debug_return_int(outlen + 1);
    }

    /*
     * Print out line with word wrap around maxlen characters.
     */
    while (linelen > maxlen) {
	end = beg + maxlen;
	while (end != beg && *end != ' ')
	    end--;
	if (beg == end) {
	    /* Unable to find word break within maxlen, look beyond. */
	    end = strchr(beg + maxlen, ' ');
	    if (end == NULL)
		break;	/* no word break */
	}
	len = fprintf(fp, "%s%.*s\n", indent, (int)(end - beg), beg);
	if (len < 0)
	    debug_return_ssize_t(-1);
	outlen += len;
	while (*end == ' ')
	    end++;
	linelen -= (end - beg);
	beg = end;
	if (indent[0] == '\0') {
	    indent = EVENTLOG_INDENT;
	    maxlen -= sizeof(EVENTLOG_INDENT) - 1;
	}
    }
    /* Print remainder, if any. */
    if (linelen) {
	len = fprintf(fp, "%s%s\n", indent, beg);
	if (len < 0)
	    debug_return_ssize_t(-1);
	outlen += len;
    }

    debug_return_size_t(outlen);
}
