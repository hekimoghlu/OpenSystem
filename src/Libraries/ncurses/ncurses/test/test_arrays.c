/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 12, 2022.
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
 * $Id: test_arrays.c,v 1.5 2010/11/13 19:57:57 tom Exp $
 *
 * Author: Thomas E Dickey
 *
 * Demonstrate the public arrays from the terminfo library.

extern NCURSES_EXPORT_VAR(NCURSES_CONST char * const ) boolnames[];
extern NCURSES_EXPORT_VAR(NCURSES_CONST char * const ) boolcodes[];
extern NCURSES_EXPORT_VAR(NCURSES_CONST char * const ) boolfnames[];
extern NCURSES_EXPORT_VAR(NCURSES_CONST char * const ) numnames[];
extern NCURSES_EXPORT_VAR(NCURSES_CONST char * const ) numcodes[];
extern NCURSES_EXPORT_VAR(NCURSES_CONST char * const ) numfnames[];
extern NCURSES_EXPORT_VAR(NCURSES_CONST char * const ) strnames[];
extern NCURSES_EXPORT_VAR(NCURSES_CONST char * const ) strcodes[];
extern NCURSES_EXPORT_VAR(NCURSES_CONST char * const ) strfnames[];

 */

#define USE_TINFO
#include <test.priv.h>

#if HAVE_TIGETSTR
#if defined(HAVE_CURSES_DATA_BOOLNAMES) || defined(DECL_CURSES_DATA_BOOLNAMES)

#define DUMP(name) dump_array(#name, name)

static void
dump_array(const char *name, NCURSES_CONST char *const *list)
{
    int n;

    printf("%s:\n", name);
    for (n = 0; list[n] != 0; ++n) {
	printf("%5d:%s\n", n, list[n]);
    }
}

int
main(int argc GCC_UNUSED, char *argv[]GCC_UNUSED)
{
    DUMP(boolnames);
    DUMP(boolcodes);
    DUMP(boolfnames);

    DUMP(numnames);
    DUMP(numcodes);
    DUMP(numfnames);

    DUMP(strnames);
    DUMP(strcodes);
    DUMP(strfnames);

    ExitProgram(EXIT_SUCCESS);
}

#else
int
main(int argc GCC_UNUSED, char *argv[]GCC_UNUSED)
{
    printf("This program requires the terminfo arrays\n");
    ExitProgram(EXIT_FAILURE);
}
#endif
#else /* !HAVE_TIGETSTR */
int
main(int argc GCC_UNUSED, char *argv[]GCC_UNUSED)
{
    printf("This program requires the terminfo functions such as tigetstr\n");
    ExitProgram(EXIT_FAILURE);
}
#endif /* HAVE_TIGETSTR */
