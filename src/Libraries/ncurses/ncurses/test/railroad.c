/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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
 * Author: Thomas E. Dickey - 2000
 *
 * $Id: railroad.c,v 1.21 2013/09/28 22:02:17 tom Exp $
 *
 * A simple demo of the termcap interface.
 */
#define USE_TINFO
#include <test.priv.h>

#if HAVE_TGETENT

static char *wipeit;
static char *moveit;
static int length;
static int height;

static char *finisC;
static char *finisS;
static char *finisU;

static char *startC;
static char *startS;
static char *startU;

static char *backup;

static bool interrupted = FALSE;

static
TPUTS_PROTO(outc, c)
{
    int rc = OK;

    if (interrupted) {
	char tmp = (char) c;
	if (write(STDOUT_FILENO, &tmp, (size_t) 1) == -1)
	    rc = ERR;
    } else {
	if (putc(c, stdout) == EOF)
	    rc = ERR;
    }
    TPUTS_RETURN(rc);
}

static void
PutChar(int ch)
{
    putchar(ch);
    fflush(stdout);
    napms(moveit ? 10 : 50);	/* not really termcap... */
}

static void
Backup(void)
{
    tputs(backup != 0 ? backup : "\b", 1, outc);
}

static void
MyShowCursor(int flag)
{
    if (startC != 0 && finisC != 0) {
	tputs(flag ? startC : finisC, 1, outc);
    }
}

static void
StandOut(int flag)
{
    if (startS != 0 && finisS != 0) {
	tputs(flag ? startS : finisS, 1, outc);
    }
}

static void
Underline(int flag)
{
    if (startU != 0 && finisU != 0) {
	tputs(flag ? startU : finisU, 1, outc);
    }
}

static void
ShowSign(char *string)
{
    char *base = string;
    int ch, first, last;

    if (moveit != 0) {
	tputs(tgoto(moveit, 0, height - 1), 1, outc);
	tputs(wipeit, 1, outc);
    }

    while (*string != 0) {
	ch = *string;
	if (ch != ' ') {
	    if (moveit != 0) {
		for (first = length - 2; first >= (string - base); first--) {
		    if (first < length - 1) {
			tputs(tgoto(moveit, first + 1, height - 1), 1, outc);
			PutChar(' ');
		    }
		    tputs(tgoto(moveit, first, height - 1), 1, outc);
		    PutChar(ch);
		}
	    } else {
		last = ch;
		if (isalpha(ch)) {
		    first = isupper(ch) ? 'A' : 'a';
		} else if (isdigit(ch)) {
		    first = '0';
		} else {
		    first = ch;
		}
		if (first < last) {
		    Underline(1);
		    while (first < last) {
			PutChar(first);
			Backup();
			first++;
		    }
		    Underline(0);
		}
	    }
	    if (moveit != 0)
		Backup();
	}
	StandOut(1);
	PutChar(ch);
	StandOut(0);
	fflush(stdout);
	string++;
    }
    if (moveit != 0)
	tputs(wipeit, 1, outc);
    putchar('\n');
}

static void
cleanup(void)
{
    Underline(0);
    StandOut(0);
    MyShowCursor(1);
}

static void
onsig(int n GCC_UNUSED)
{
    interrupted = TRUE;
    cleanup();
    ExitProgram(EXIT_FAILURE);
}

static void
railroad(char **args)
{
    NCURSES_CONST char *name = getenv("TERM");
    char buffer[1024];
    char area[1024], *ap = area;

    if (name == 0)
	name = "dumb";
    if (tgetent(buffer, name) >= 0) {

	wipeit = tgetstr("ce", &ap);
	height = tgetnum("li");
	length = tgetnum("co");
	moveit = tgetstr("cm", &ap);

	if (wipeit == 0
	    || moveit == 0
	    || height <= 0
	    || length <= 0) {
	    wipeit = 0;
	    moveit = 0;
	    height = 0;
	    length = 0;
	}

	startS = tgetstr("so", &ap);
	finisS = tgetstr("se", &ap);

	startU = tgetstr("us", &ap);
	finisU = tgetstr("ue", &ap);

	backup = tgetstr("le", &ap);

	startC = tgetstr("ve", &ap);
	finisC = tgetstr("vi", &ap);

	MyShowCursor(0);

	CATCHALL(onsig);

	while (*args) {
	    ShowSign(*args++);
	}
	MyShowCursor(1);
    }
}

int
main(int argc, char *argv[])
{
    if (argc > 1) {
	railroad(argv + 1);
    } else {
	static char world[] = "Hello World";
	static char *hello[] =
	{world, 0};
	railroad(hello);
    }
    ExitProgram(EXIT_SUCCESS);
}

#else
int
main(int argc GCC_UNUSED,
     char *argv[]GCC_UNUSED)
{
    printf("This program requires termcap\n");
    exit(EXIT_FAILURE);
}
#endif
