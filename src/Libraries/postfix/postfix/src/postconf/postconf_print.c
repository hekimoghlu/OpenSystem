/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 24, 2022.
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

#include <sys_defs.h>
#include <string.h>
#include <stdarg.h>

/* Utility library. */

#include <msg.h>
#include <vstream.h>
#include <vstring.h>

/* Application-specific. */

#include <postconf.h>

/* SLMs. */

#define STR(x)	vstring_str(x)

/* pcf_print_line - show line possibly folded, and with normalized whitespace */

void    pcf_print_line(VSTREAM *fp, int mode, const char *fmt,...)
{
    va_list ap;
    static VSTRING *buf = 0;
    char   *start;
    char   *next;
    int     line_len = 0;
    int     word_len;

    /*
     * One-off initialization.
     */
    if (buf == 0)
	buf = vstring_alloc(100);

    /*
     * Format the text.
     */
    va_start(ap, fmt);
    vstring_vsprintf(buf, fmt, ap);
    va_end(ap);

    /*
     * Normalize the whitespace. We don't use the line_wrap() routine because
     * 1) that function does not normalize whitespace between words and 2) we
     * want to normalize whitespace even when not wrapping lines.
     * 
     * XXX Some parameters preserve whitespace: for example, smtpd_banner and
     * smtpd_reject_footer. If we have to preserve whitespace between words,
     * then perhaps readlline() can be changed to canonicalize whitespace
     * that follows a newline.
     */
    for (start = STR(buf); *(start += strspn(start, PCF_SEPARATORS)) != 0; start = next) {
	word_len = strcspn(start, PCF_SEPARATORS);
	if (*(next = start + word_len) != 0)
	    *next++ = 0;
	if (word_len > 0 && line_len > 0) {
	    if ((mode & PCF_FOLD_LINE) == 0
		|| line_len + word_len < PCF_LINE_LIMIT) {
		vstream_fputs(" ", fp);
		line_len += 1;
	    } else {
		vstream_fputs("\n" PCF_INDENT_TEXT, fp);
		line_len = PCF_INDENT_LEN;
	    }
	}
	vstream_fputs(start, fp);
	line_len += word_len;
    }
    vstream_fputs("\n", fp);
}
