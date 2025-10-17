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
/****************************************************************************
 *  Author: Thomas Dickey 2002                                              *
 ****************************************************************************/

#include <curses.priv.h>

MODULE_ID("$Id: lib_wacs.c,v 1.14 2014/02/23 01:21:08 tom Exp $")

NCURSES_EXPORT_VAR(cchar_t) * _nc_wacs = 0;

NCURSES_EXPORT(void)
_nc_init_wacs(void)
{
    /* *INDENT-OFF* */
    static const struct {
	unsigned map;
	int	value[2];
    } table[] = {
	/* VT100 symbols */
	{ 'l',	{ '+',	0x250c }},	/* upper left corner */
	{ 'm',	{ '+',	0x2514 }},	/* lower left corner */
	{ 'k',	{ '+',	0x2510 }},	/* upper right corner */
	{ 'j',	{ '+',	0x2518 }},	/* lower right corner */
	{ 't',	{ '+',	0x251c }},	/* tee pointing left */
	{ 'u',	{ '+',	0x2524 }},	/* tee pointing right */
	{ 'v',	{ '+',	0x2534 }},	/* tee pointing up */
	{ 'w',	{ '+',	0x252c }},	/* tee pointing down */
	{ 'q',	{ '-',	0x2500 }},	/* horizontal line */
	{ 'x',	{ '|',	0x2502 }},	/* vertical line */
	{ 'n',	{ '+',	0x253c }},	/* large plus or crossover */
	{ 'o',	{ '~',	0x23ba }},	/* scan line 1 */
	{ 's',	{ '_',	0x23bd }},	/* scan line 9 */
	{ '`',	{ '+',	0x25c6 }},	/* diamond */
	{ 'a',	{ ':',	0x2592 }},	/* checker board (stipple) */
	{ 'f',	{ '\'',	0x00b0 }},	/* degree symbol */
	{ 'g',	{ '#',	0x00b1 }},	/* plus/minus */
	{ '~',	{ 'o',	0x00b7 }},	/* bullet */
	/* Teletype 5410v1 symbols */
	{ ',',	{ '<',	0x2190 }},	/* arrow pointing left */
	{ '+',	{ '>',	0x2192 }},	/* arrow pointing right */
	{ '.',	{ 'v',	0x2193 }},	/* arrow pointing down */
	{ '-',	{ '^',	0x2191 }},	/* arrow pointing up */
	{ 'h',	{ '#',	0x2592 }},	/* board of squares */
	{ 'i',	{ '#',	0x2603 }},	/* lantern symbol */
	{ '0',	{ '#',	0x25ae }},	/* solid square block */
	/* these defaults were invented for ncurses */
	{ 'p',	{ '-',	0x23bb }},	/* scan line 3 */
	{ 'r',	{ '-',	0x23bc }},	/* scan line 7 */
	{ 'y',	{ '<',	0x2264 }},	/* less-than-or-equal-to */
	{ 'z',	{ '>',	0x2265 }},	/* greater-than-or-equal-to */
	{ '{',	{ '*',	0x03c0 }},	/* greek pi */
	{ '|',	{ '!',	0x2260 }},	/* not-equal */
	{ '}',	{ 'f',	0x00a3 }},	/* pound-sterling symbol */
	/* thick-line-drawing */
	{ 'L',	{ '+',	0x250f }},	/* upper left corner */
	{ 'M',	{ '+',	0x2517 }},	/* lower left corner */
	{ 'K',	{ '+',	0x2513 }},	/* upper right corner */
	{ 'J',	{ '+',	0x251b }},	/* lower right corner */
	{ 'T',	{ '+',	0x2523 }},	/* tee pointing left */
	{ 'U',	{ '+',	0x252b }},	/* tee pointing right */
	{ 'V',	{ '+',	0x253b }},	/* tee pointing up */
	{ 'W',	{ '+',	0x2533 }},	/* tee pointing down */
	{ 'Q',	{ '-',	0x2501 }},	/* horizontal line */
	{ 'X',	{ '|',	0x2503 }},	/* vertical line */
	{ 'N',	{ '+',	0x254b }},	/* large plus or crossover */
	/* double-line-drawing */
	{ 'C',	{ '+',	0x2554 }},	/* upper left corner */
	{ 'D',	{ '+',	0x255a }},	/* lower left corner */
	{ 'B',	{ '+',	0x2557 }},	/* upper right corner */
	{ 'A',	{ '+',	0x255d }},	/* lower right corner */
	{ 'G',	{ '+',	0x2563 }},	/* tee pointing left */
	{ 'F',	{ '+',	0x2560 }},	/* tee pointing right */
	{ 'H',	{ '+',	0x2569 }},	/* tee pointing up */
	{ 'I',	{ '+',	0x2566 }},	/* tee pointing down */
	{ 'R',	{ '-',	0x2550 }},	/* horizontal line */
	{ 'Y',	{ '|',	0x2551 }},	/* vertical line */
	{ 'E',	{ '+',	0x256c }},	/* large plus or crossover */
    };
    /* *INDENT-ON* */

    unsigned n, m;
    int active = _nc_unicode_locale();

    /*
     * If we're running in a UTF-8 locale, will use the Unicode equivalents
     * rather than the terminfo information.  Actually the terminfo should
     * be the rule, but there are people who are offended by the notion that
     * a Unicode-capable terminal would have something resembling a mode.
     * So the smacs/rmacs may be disabled -- sometime.
     */
    T(("initializing WIDE-ACS map (Unicode is%s active)",
       active ? "" : " not"));

    if ((_nc_wacs = typeCalloc(cchar_t, ACS_LEN)) != 0) {

	for (n = 0; n < SIZEOF(table); ++n) {
	    int wide = wcwidth((wchar_t) table[n].value[active]);

	    m = table[n].map;
	    if (active && (wide == 1)) {
		SetChar(_nc_wacs[m], table[n].value[1], A_NORMAL);
	    } else if (acs_map[m] & A_ALTCHARSET) {
		SetChar(_nc_wacs[m], m, A_ALTCHARSET);
	    } else {
		SetChar(_nc_wacs[m], table[n].value[0], A_NORMAL);
	    }

	    T(("#%d, SetChar(%c, %#04x) = %s",
	       n, m,
	       table[n].value[active],
	       _tracecchar_t(&_nc_wacs[m])));
	}
    }
}
