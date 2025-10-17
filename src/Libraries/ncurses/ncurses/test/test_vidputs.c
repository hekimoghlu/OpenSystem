/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 2, 2024.
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
 * $Id: test_vidputs.c,v 1.5 2014/07/19 23:09:58 tom Exp $
 *
 * Demonstrate the vidputs and vidattr functions.
 * Thomas Dickey - 2013/01/12
 */

#define USE_TINFO
#include <test.priv.h>

#if HAVE_SETUPTERM && HAVE_VIDPUTS

#define valid(s) ((s != 0) && s != (char *)-1)

static FILE *my_fp;
static bool p_opt = FALSE;

static
TPUTS_PROTO(outc, c)
{
    int rc = c;

    rc = putc(c, my_fp);
    TPUTS_RETURN(rc);
}

static bool
outs(const char *s)
{
    if (valid(s)) {
	tputs(s, 1, outc);
	return TRUE;
    }
    return FALSE;
}

static void
cleanup(void)
{
    outs(exit_attribute_mode);
    if (!outs(orig_colors))
	outs(orig_pair);
    outs(cursor_normal);
}

static void
change_attr(chtype attr)
{
    if (p_opt) {
	vidputs(attr, outc);
    } else {
	vidattr(attr);
    }
}

static void
test_vidputs(void)
{
    fprintf(my_fp, "Name: ");
    change_attr(A_BOLD);
    fputs("Bold", my_fp);
    change_attr(A_REVERSE);
    fputs(" Reverse", my_fp);
    change_attr(A_NORMAL);
    fputs("\n", my_fp);
}

static void
usage(void)
{
    static const char *tbl[] =
    {
	"Usage: test_vidputs [options]"
	,""
	,"Options:"
	,"  -e      use stderr (default stdout)"
	,"  -p      use vidputs (default vidattr)"
    };
    unsigned n;
    for (n = 0; n < SIZEOF(tbl); ++n)
	fprintf(stderr, "%s\n", tbl[n]);
    ExitProgram(EXIT_FAILURE);
}

int
main(int argc GCC_UNUSED, char *argv[]GCC_UNUSED)
{
    int ch;

    my_fp = stdout;

    while ((ch = getopt(argc, argv, "ep")) != -1) {
	switch (ch) {
	case 'e':
	    my_fp = stderr;
	    break;
	case 'p':
	    p_opt = TRUE;
	    break;
	default:
	    usage();
	    break;
	}
    }
    if (optind < argc)
	usage();

    setupterm((char *) 0, 1, (int *) 0);
    test_vidputs();
    cleanup();
    ExitProgram(EXIT_SUCCESS);
}
#else
int
main(int argc GCC_UNUSED,
     char *argv[]GCC_UNUSED)
{
    fprintf(stderr, "This program requires terminfo\n");
    exit(EXIT_FAILURE);
}
#endif
