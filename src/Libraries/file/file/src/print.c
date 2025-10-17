/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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
 * print.c - debugging printout routines
 */

#include "file.h"

#ifndef lint
FILE_RCSID("@(#)$File: print.c,v 1.89 2021/06/30 10:08:48 christos Exp $")
#endif  /* lint */

#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <time.h>

#include "cdf.h"

#ifndef COMPILE_ONLY
protected void
file_mdump(struct magic *m)
{
	static const char optyp[] = { FILE_OPS };
	char tbuf[256];

	(void) fprintf(stderr, "%u: %.*s %u", m->lineno,
	    (m->cont_level & 7) + 1, ">>>>>>>>", m->offset);

	if (m->flag & INDIR) {
		(void) fprintf(stderr, "(%s,",
		    /* Note: type is unsigned */
		    (m->in_type < file_nnames) ? file_names[m->in_type] :
		    "*bad in_type*");
		if (m->in_op & FILE_OPINVERSE)
			(void) fputc('~', stderr);
		(void) fprintf(stderr, "%c%u),",
		    (CAST(size_t, m->in_op & FILE_OPS_MASK) <
		    __arraycount(optyp)) ?
		    optyp[m->in_op & FILE_OPS_MASK] : '?', m->in_offset);
	}
	(void) fprintf(stderr, " %s%s", (m->flag & UNSIGNED) ? "u" : "",
	    /* Note: type is unsigned */
	    (m->type < file_nnames) ? file_names[m->type] : "*bad type");
	if (m->mask_op & FILE_OPINVERSE)
		(void) fputc('~', stderr);

	if (IS_STRING(m->type)) {
		if (m->str_flags) {
			(void) fputc('/', stderr);
			if (m->str_flags & STRING_COMPACT_WHITESPACE)
				(void) fputc(CHAR_COMPACT_WHITESPACE, stderr);
			if (m->str_flags & STRING_COMPACT_OPTIONAL_WHITESPACE)
				(void) fputc(CHAR_COMPACT_OPTIONAL_WHITESPACE,
				    stderr);
			if (m->str_flags & STRING_IGNORE_LOWERCASE)
				(void) fputc(CHAR_IGNORE_LOWERCASE, stderr);
			if (m->str_flags & STRING_IGNORE_UPPERCASE)
				(void) fputc(CHAR_IGNORE_UPPERCASE, stderr);
			if (m->str_flags & REGEX_OFFSET_START)
				(void) fputc(CHAR_REGEX_OFFSET_START, stderr);
			if (m->str_flags & STRING_TEXTTEST)
				(void) fputc(CHAR_TEXTTEST, stderr);
			if (m->str_flags & STRING_BINTEST)
				(void) fputc(CHAR_BINTEST, stderr);
			if (m->str_flags & PSTRING_1_BE)
				(void) fputc(CHAR_PSTRING_1_BE, stderr);
			if (m->str_flags & PSTRING_2_BE)
				(void) fputc(CHAR_PSTRING_2_BE, stderr);
			if (m->str_flags & PSTRING_2_LE)
				(void) fputc(CHAR_PSTRING_2_LE, stderr);
			if (m->str_flags & PSTRING_4_BE)
				(void) fputc(CHAR_PSTRING_4_BE, stderr);
			if (m->str_flags & PSTRING_4_LE)
				(void) fputc(CHAR_PSTRING_4_LE, stderr);
			if (m->str_flags & PSTRING_LENGTH_INCLUDES_ITSELF)
				(void) fputc(
				    CHAR_PSTRING_LENGTH_INCLUDES_ITSELF,
				    stderr);
		}
		if (m->str_range)
			(void) fprintf(stderr, "/%u", m->str_range);
	}
	else {
		if (CAST(size_t, m->mask_op & FILE_OPS_MASK) <
		    __arraycount(optyp))
			(void) fputc(optyp[m->mask_op & FILE_OPS_MASK], stderr);
		else
			(void) fputc('?', stderr);

		if (m->num_mask) {
			(void) fprintf(stderr, "%.8llx",
			    CAST(unsigned long long, m->num_mask));
		}
	}
	(void) fprintf(stderr, ",%c", m->reln);

	if (m->reln != 'x') {
		switch (m->type) {
		case FILE_BYTE:
		case FILE_SHORT:
		case FILE_LONG:
		case FILE_LESHORT:
		case FILE_LELONG:
		case FILE_MELONG:
		case FILE_BESHORT:
		case FILE_BELONG:
		case FILE_INDIRECT:
			(void) fprintf(stderr, "%d", m->value.l);
			break;
		case FILE_BEQUAD:
		case FILE_LEQUAD:
		case FILE_QUAD:
		case FILE_OFFSET:
			(void) fprintf(stderr, "%" INT64_T_FORMAT "d",
			    CAST(long long, m->value.q));
			break;
		case FILE_PSTRING:
		case FILE_STRING:
		case FILE_REGEX:
		case FILE_BESTRING16:
		case FILE_LESTRING16:
		case FILE_SEARCH:
			file_showstr(stderr, m->value.s,
			    CAST(size_t, m->vallen));
			break;
		case FILE_DATE:
		case FILE_LEDATE:
		case FILE_BEDATE:
		case FILE_MEDATE:
			(void)fprintf(stderr, "%s,",
			    file_fmttime(tbuf, sizeof(tbuf), m->value.l, 0));
			break;
		case FILE_LDATE:
		case FILE_LELDATE:
		case FILE_BELDATE:
		case FILE_MELDATE:
			(void)fprintf(stderr, "%s,",
			    file_fmttime(tbuf, sizeof(tbuf), m->value.l,
			    FILE_T_LOCAL));
			break;
		case FILE_QDATE:
		case FILE_LEQDATE:
		case FILE_BEQDATE:
			(void)fprintf(stderr, "%s,",
			    file_fmttime(tbuf, sizeof(tbuf), m->value.q, 0));
			break;
		case FILE_QLDATE:
		case FILE_LEQLDATE:
		case FILE_BEQLDATE:
			(void)fprintf(stderr, "%s,",
			    file_fmttime(tbuf, sizeof(tbuf), m->value.q,
			    FILE_T_LOCAL));
			break;
		case FILE_QWDATE:
		case FILE_LEQWDATE:
		case FILE_BEQWDATE:
			(void)fprintf(stderr, "%s,",
			    file_fmttime(tbuf, sizeof(tbuf), m->value.q,
			    FILE_T_WINDOWS));
			break;
		case FILE_FLOAT:
		case FILE_BEFLOAT:
		case FILE_LEFLOAT:
			(void) fprintf(stderr, "%G", m->value.f);
			break;
		case FILE_DOUBLE:
		case FILE_BEDOUBLE:
		case FILE_LEDOUBLE:
			(void) fprintf(stderr, "%G", m->value.d);
			break;
		case FILE_LEVARINT:
		case FILE_BEVARINT:
			(void)fprintf(stderr, "%s", file_fmtvarint(
			    m->value.us, m->type, tbuf, sizeof(tbuf)));
			break;
		case FILE_DEFAULT:
			/* XXX - do anything here? */
			break;
		case FILE_USE:
		case FILE_NAME:
		case FILE_DER:
			(void) fprintf(stderr, "'%s'", m->value.s);
			break;
		case FILE_GUID:
			(void) file_print_guid(tbuf, sizeof(tbuf),
			    m->value.guid);
			(void) fprintf(stderr, "%s", tbuf);
			break;

		default:
			(void) fprintf(stderr, "*bad type %d*", m->type);
			break;
		}
	}
	(void) fprintf(stderr, ",\"%s\"]\n", m->desc);
}
#endif

/*VARARGS*/
protected void
file_magwarn(struct magic_set *ms, const char *f, ...)
{
	va_list va;

	/* cuz we use stdout for most, stderr here */
	(void) fflush(stdout);

	if (ms->file)
		(void) fprintf(stderr, "%s, %lu: ", ms->file,
		    CAST(unsigned long, ms->line));
	(void) fprintf(stderr, "Warning: ");
	va_start(va, f);
	(void) vfprintf(stderr, f, va);
	va_end(va);
	(void) fputc('\n', stderr);
}

protected const char *
file_fmtvarint(const unsigned char *us, int t, char *buf, size_t blen)
{
	snprintf(buf, blen, "%jd", file_varint2uintmax_t(us, t, NULL));
	return buf;
}

protected const char *
file_fmttime(char *buf, size_t bsize, uint64_t v, int flags)
{
	char *pp;
	time_t t;
	struct tm *tm, tmz;

	if (flags & FILE_T_WINDOWS) {
		struct timespec ts;
		cdf_timestamp_to_timespec(&ts, CAST(cdf_timestamp_t, v));
		t = ts.tv_sec;
	} else {
		// XXX: perhaps detect and print something if overflow
		// on 32 bit time_t?
		t = CAST(time_t, v);
	}

	if (flags & FILE_T_LOCAL) {
		tm = localtime_r(&t, &tmz);
	} else {
		tm = gmtime_r(&t, &tmz);
	}
	if (tm == NULL)
		goto out;
	pp = asctime_r(tm, buf);

	if (pp == NULL)
		goto out;
	pp[strcspn(pp, "\n")] = '\0';
	return pp;
out:
	strlcpy(buf, "*Invalid time*", bsize);
	return buf;
}
