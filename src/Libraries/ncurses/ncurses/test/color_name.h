/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
 * $Id: color_name.h,v 1.4 2012/11/18 01:59:32 tom Exp $
 */

#ifndef __COLORNAME_H
#define __COLORNAME_H 1

#ifndef __TEST_PRIV_H
#include <test.priv.h>
#endif

static NCURSES_CONST char *the_color_names[] =
{
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
    "BLACK",
    "RED",
    "GREEN",
    "YELLOW",
    "BLUE",
    "MAGENTA",
    "CYAN",
    "WHITE"
};

#ifdef NEED_COLOR_CODE
static int
color_code(const char *color)
{
    int result = 0;
    char *endp = 0;
    size_t n;

    if ((result = (int) strtol(color, &endp, 0)) >= 0
	&& (endp == 0 || *endp == 0)) {
	;
    } else if (!strcmp(color, "default")) {
	result = -1;
    } else {
	for (n = 0; n < SIZEOF(the_color_names); ++n) {
	    if (!strcmp(the_color_names[n], color)) {
		result = (int) n;
		break;
	    }
	}
    }
    return result;
}
#endif /* NEED_COLOR_NAME */

#ifdef NEED_COLOR_NAME
static const char *
color_name(int color)
{
    static char temp[20];
    const char *result = 0;

    if (color >= (int) SIZEOF(the_color_names)) {
	sprintf(temp, "%d", color);
	result = temp;
    } else if (color < 0) {
	result = "default";
    } else {
	result = the_color_names[color];
    }
    return result;
}
#endif /* NEED_COLOR_NAME */

#endif /* __COLORNAME_H */
