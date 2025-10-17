/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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
 * CHARMAP file handling for localedef.
 */
#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/tree.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stddef.h>
#include <unistd.h>
#include "localedef.h"
#include "parser.h"


typedef struct charmap {
	const char *name;
	wchar_t wc;
	RB_ENTRY(charmap) rb_sym;
	RB_ENTRY(charmap) rb_wc;
} charmap_t;

static int cmap_compare_sym(const void *n1, const void *n2);
static int cmap_compare_wc(const void *n1, const void *n2);

static RB_HEAD(cmap_sym, charmap) cmap_sym;
static RB_HEAD(cmap_wc, charmap) cmap_wc;

RB_GENERATE_STATIC(cmap_sym, charmap, rb_sym, cmap_compare_sym);
RB_GENERATE_STATIC(cmap_wc, charmap, rb_wc, cmap_compare_wc);

/*
 * Array of POSIX specific portable characters.
 */

static const struct {
	const char *name;
	int	ch;
} portable_chars[] = {
	{ "NUL",		'\0' },
	{ "SOH",		'\x01' },
	{ "STX",		'\x02' },
	{ "ETX",		'\x03' },
	{ "EOT",		'\x04' },
	{ "ENQ",		'\x05' },
	{ "ACK",		'\x06' },
	{ "BEL",		'\a' },
	{ "alert",		'\a' },
	{ "BS",			'\b' },
	{ "backspace",		'\b' },
	{ "HT",			'\t' },
	{ "tab",		'\t' },
	{ "LF",			'\n' },
	{ "newline",		'\n' },
	{ "VT",			'\v' },
	{ "vertical-tab",	'\v' },
	{ "FF",			'\f' },
	{ "form-feed",		'\f' },
	{ "CR",			'\r' },
	{ "carriage-return",	'\r' },
	{ "SO",			'\x0e' },
	{ "SI",			'\x0f' },
	{ "DLE",		'\x10' },
	{ "DC1",		'\x11' },
	{ "DC2",		'\x12' },
	{ "DC3",		'\x13' },
	{ "DC4",		'\x14' },
	{ "NAK",		'\x15' },
	{ "SYN",		'\x16' },
	{ "ETB",		'\x17' },
	{ "CAN",		'\x18' },
	{ "EM",			'\x19' },
	{ "SUB",		'\x1a' },
	{ "ESC",		'\x1b' },
	{ "FS",			'\x1c' },
	{ "IS4",		'\x1c' },
	{ "GS",			'\x1d' },
	{ "IS3",		'\x1d' },
	{ "RS",			'\x1e' },
	{ "IS2",		'\x1e' },
	{ "US",			'\x1f' },
	{ "IS1",		'\x1f' },
	{ "DEL",		'\x7f' },
	{ "space",		' ' },
	{ "exclamation-mark",	'!' },
	{ "quotation-mark",	'"' },
	{ "number-sign",	'#' },
	{ "dollar-sign",	'$' },
	{ "percent-sign",	'%' },
	{ "ampersand",		'&' },
	{ "apostrophe",		'\'' },
	{ "left-parenthesis",	'(' },
	{ "right-parenthesis",	')' },
	{ "asterisk",		'*' },
	{ "plus-sign",		'+' },
	{ "comma",		 ','},
	{ "hyphen-minus",	'-' },
	{ "hyphen",		'-' },
	{ "full-stop",		'.' },
	{ "period",		'.' },
	{ "slash",		'/' },
	{ "solidus",		'/' },
	{ "zero",		'0' },
	{ "one",		'1' },
	{ "two",		'2' },
	{ "three",		'3' },
	{ "four",		'4' },
	{ "five",		'5' },
	{ "six",		'6' },
	{ "seven",		'7' },
	{ "eight",		'8' },
	{ "nine",		'9' },
	{ "colon",		':' },
	{ "semicolon",		';' },
#ifdef __APPLE__
	{ "less-then-sign",	'<' },
#endif
	{ "less-than-sign",	'<' },
	{ "equals-sign",	'=' },
#ifdef __APPLE__
	{ "greater-then-sign",	'>' },
#endif
	{ "greater-than-sign",	'>' },
	{ "question-mark",	'?' },
	{ "commercial-at",	'@' },
	{ "left-square-bracket", '[' },
	{ "backslash",		'\\' },
	{ "reverse-solidus",	'\\' },
	{ "right-square-bracket", ']' },
	{ "circumflex",		'^' },
	{ "circumflex-accent",	'^' },
	{ "low-line",		'_' },
	{ "underscore",		'_' },
#ifdef __APPLE__
	{ "underline",		'_' },
#endif
	{ "grave-accent",	'`' },
	{ "left-brace",		'{' },
	{ "left-curly-bracket",	'{' },
	{ "vertical-line",	'|' },
	{ "right-brace",	'}' },
	{ "right-curly-bracket", '}' },
	{ "tilde",		'~' },
	{ "A", 'A' },
	{ "B", 'B' },
	{ "C", 'C' },
	{ "D", 'D' },
	{ "E", 'E' },
	{ "F", 'F' },
	{ "G", 'G' },
	{ "H", 'H' },
	{ "I", 'I' },
	{ "J", 'J' },
	{ "K", 'K' },
	{ "L", 'L' },
	{ "M", 'M' },
	{ "N", 'N' },
	{ "O", 'O' },
	{ "P", 'P' },
	{ "Q", 'Q' },
	{ "R", 'R' },
	{ "S", 'S' },
	{ "T", 'T' },
	{ "U", 'U' },
	{ "V", 'V' },
	{ "W", 'W' },
	{ "X", 'X' },
	{ "Y", 'Y' },
	{ "Z", 'Z' },
	{ "a", 'a' },
	{ "b", 'b' },
	{ "c", 'c' },
	{ "d", 'd' },
	{ "e", 'e' },
	{ "f", 'f' },
	{ "g", 'g' },
	{ "h", 'h' },
	{ "i", 'i' },
	{ "j", 'j' },
	{ "k", 'k' },
	{ "l", 'l' },
	{ "m", 'm' },
	{ "n", 'n' },
	{ "o", 'o' },
	{ "p", 'p' },
	{ "q", 'q' },
	{ "r", 'r' },
	{ "s", 's' },
	{ "t", 't' },
	{ "u", 'u' },
	{ "v", 'v' },
	{ "w", 'w' },
	{ "x", 'x' },
	{ "y", 'y' },
	{ "z", 'z' },
	{ NULL, 0 }
};

static int
cmap_compare_sym(const void *n1, const void *n2)
{
	const charmap_t *c1 = n1;
	const charmap_t *c2 = n2;
	int rv;

	rv = strcmp(c1->name, c2->name);
	return ((rv < 0) ? -1 : (rv > 0) ? 1 : 0);
}

static int
cmap_compare_wc(const void *n1, const void *n2)
{
	const charmap_t *c1 = n1;
	const charmap_t *c2 = n2;

	return ((c1->wc < c2->wc) ? -1 : (c1->wc > c2->wc) ? 1 : 0);
}

void
init_charmap(void)
{
	RB_INIT(&cmap_sym);

	RB_INIT(&cmap_wc);
}

static void
add_charmap_impl(const char *sym, wchar_t wc, int nodups)
{
	charmap_t	srch;
	charmap_t	*n = NULL;

	srch.wc = wc;
	srch.name = sym;

	/*
	 * also possibly insert the wide mapping, although note that there
	 * can only be one of these per wide character code.
	 */
	if ((wc != (wchar_t)-1) && ((RB_FIND(cmap_wc, &cmap_wc, &srch)) == NULL)) {
		if ((n = calloc(1, sizeof (*n))) == NULL) {
			errf("out of memory");
			return;
		}
		n->wc = wc;
		RB_INSERT(cmap_wc, &cmap_wc, n);
	}

	if (sym) {
		if (RB_FIND(cmap_sym, &cmap_sym, &srch) != NULL) {
			if (nodups) {
				errf("duplicate character definition");
			}
			return;
		}
		if ((n == NULL) && ((n = calloc(1, sizeof (*n))) == NULL)) {
			errf("out of memory");
			return;
		}
		n->wc = wc;
		n->name = sym;

		RB_INSERT(cmap_sym, &cmap_sym, n);
	}
}

void
add_charmap(const char *sym, int c)
{
	add_charmap_impl(sym, c, 1);
}

void
add_charmap_undefined(char *sym)
{
	charmap_t srch;
	charmap_t *cm = NULL;

	srch.name = sym;
	cm = RB_FIND(cmap_sym, &cmap_sym, &srch);

	if ((undefok == 0) && ((cm == NULL) || (cm->wc == (wchar_t)-1))) {
		warn("undefined symbol <%s>", sym);
		add_charmap_impl(sym, -1, 0);
	} else {
		free(sym);
	}
}

void
add_charmap_range(char *s, char *e, int wc)
{
	int	ls, le;
	int	si;
	int	sn, en;
	int	i;

	static const char *digits = "0123456789";

	ls = strlen(s);
	le = strlen(e);

	if (((si = strcspn(s, digits)) == 0) || (si == ls) ||
	    (strncmp(s, e, si) != 0) ||
	    ((int)strspn(s + si, digits) != (ls - si)) ||
	    ((int)strspn(e + si, digits) != (le - si)) ||
	    ((sn = atoi(s + si)) > ((en = atoi(e + si))))) {
		errf("malformed charmap range");
		return;
	}

	s[si] = 0;

	for (i = sn; i <= en; i++) {
		char *nn;
		(void) asprintf(&nn, "%s%0*u", s, ls - si, i);
		if (nn == NULL) {
			errf("out of memory");
			return;
		}

		add_charmap_impl(nn, wc, 1);
		wc++;
	}
	free(s);
	free(e);
}

void
add_charmap_char(const char *name, int val)
{
	add_charmap_impl(name, val, 0);
}

/*
 * POSIX insists that certain entries be present, even when not in the
 * original charmap file.
 */
void
add_charmap_posix(void)
{
	int	i;

	for (i = 0; portable_chars[i].name; i++) {
		add_charmap_char(portable_chars[i].name, portable_chars[i].ch);
	}
}

int
lookup_charmap(const char *sym, wchar_t *wc)
{
	charmap_t	srch;
	charmap_t	*n;

	srch.name = sym;
	n = RB_FIND(cmap_sym, &cmap_sym, &srch);
	if (n && n->wc != (wchar_t)-1) {
		if (wc)
			*wc = n->wc;
		return (0);
	}
	return (-1);
}

int
check_charmap(wchar_t wc)
{
	charmap_t srch;

	srch.wc = wc;
	return (RB_FIND(cmap_wc, &cmap_wc, &srch) ? 0 : -1);
}
