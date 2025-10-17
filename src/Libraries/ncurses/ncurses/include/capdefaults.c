/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 30, 2022.
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
 *  Author: Zeyd M. Ben-Halim <zmbenhal@netcom.com> 1992,1995               *
 *     and: Eric S. Raymond <esr@snark.thyrsus.com>                         *
 *     and: Thomas E. Dickey                        1996-on                 *
 ****************************************************************************/

/* $Id: capdefaults.c,v 1.14 2008/11/16 00:19:59 juergen Exp $ */

    /*
     * Compute obsolete capabilities.  The reason this is an include file is
     * that the two places where it's needed want the macros to generate
     * offsets to different structures.  See the file Caps for explanations of
     * these conversions.
     *
     * Note:  This code is the functional inverse of the first part of
     * postprocess_termcap().
     */
{
    char *strp;
    short capval;

#define EXTRACT_DELAY(str) \
    	(short) (strp = strchr(str, '*'), strp ? atoi(strp+1) : 0)

    /* current (4.4BSD) capabilities marked obsolete */
    if (VALID_STRING(carriage_return)
	&& (capval = EXTRACT_DELAY(carriage_return)))
	carriage_return_delay = capval;
    if (VALID_STRING(newline) && (capval = EXTRACT_DELAY(newline)))
	new_line_delay = capval;

    /* current (4.4BSD) capabilities not obsolete */
    if (!VALID_STRING(termcap_init2) && VALID_STRING(init_3string)) {
	termcap_init2 = init_3string;
	init_3string = ABSENT_STRING;
    }
    if (!VALID_STRING(termcap_reset)
     && VALID_STRING(reset_2string)
     && !VALID_STRING(reset_1string)
     && !VALID_STRING(reset_3string)) {
	termcap_reset = reset_2string;
	reset_2string = ABSENT_STRING;
    }
    if (magic_cookie_glitch_ul == ABSENT_NUMERIC
	&& magic_cookie_glitch != ABSENT_NUMERIC
	&& VALID_STRING(enter_underline_mode))
	magic_cookie_glitch_ul = magic_cookie_glitch;

    /* totally obsolete capabilities */
    linefeed_is_newline = (char) (VALID_STRING(newline)
				  && (strcmp("\n", newline) == 0));
    if (VALID_STRING(cursor_left)
	&& (capval = EXTRACT_DELAY(cursor_left)))
	backspace_delay = capval;
    if (VALID_STRING(tab) && (capval = EXTRACT_DELAY(tab)))
	horizontal_tab_delay = capval;
#undef EXTRACT_DELAY
}
