/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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
#ifndef __APPLE__
#ifndef lint
#if 0
static char sccsid[] = "@(#)from: init.c	8.1 (Berkeley) 6/4/93";
#endif
static const char rcsid[] =
    "$FreeBSD$";
#endif /* not lint */
#endif /* !__APPLE__ */

/*
 * Getty table initializations.
 *
 * Melbourne getty.
 */
#include <stdio.h>
#include <termios.h>
#include "gettytab.h"
#include "extern.h"
#include "pathnames.h"

static char loginmsg[] = "login: ";
static char nullstr[] = "";
static char loginprg[] = _PATH_LOGIN;
static char datefmt[] = "%+";

#define M(a) (char *)(&omode.c_cc[a])

struct	gettystrs gettystrs[] = {
	{ "nx", NULL, NULL },		/* next table */
	{ "cl", NULL, NULL },		/* screen clear characters */
	{ "im", NULL, NULL },		/* initial message */
	{ "lm", loginmsg, NULL },	/* login message */
	{ "er", M(VERASE), NULL },	/* erase character */
	{ "kl", M(VKILL), NULL },	/* kill character */
	{ "et", M(VEOF), NULL },	/* eof chatacter (eot) */
	{ "pc", nullstr, NULL },	/* pad character */
	{ "tt", NULL, NULL },		/* terminal type */
	{ "ev", NULL, NULL },		/* environment */
	{ "lo", loginprg, NULL },	/* login program */
	{ "hn", hostname, NULL },	/* host name */
	{ "he", NULL, NULL },		/* host name edit */
	{ "in", M(VINTR), NULL },	/* interrupt char */
	{ "qu", M(VQUIT), NULL },	/* quit char */
	{ "xn", M(VSTART), NULL },	/* XON (start) char */
	{ "xf", M(VSTOP), NULL },	/* XOFF (stop) char */
	{ "bk", M(VEOL), NULL },	/* brk char (alt \n) */
	{ "su", M(VSUSP), NULL },	/* suspend char */
	{ "ds", M(VDSUSP), NULL },	/* delayed suspend */
	{ "rp", M(VREPRINT), NULL },	/* reprint char */
	{ "fl", M(VDISCARD), NULL },	/* flush output */
	{ "we", M(VWERASE), NULL },	/* word erase */
	{ "ln", M(VLNEXT), NULL },	/* literal next */
	{ "Lo", NULL, NULL },		/* locale for strftime() */
	{ "pp", NULL, NULL },		/* ppp login program */
	{ "if", NULL, NULL },		/* sysv-like 'issue' filename */
	{ "ic", NULL, NULL },		/* modem init-chat */
	{ "ac", NULL, NULL },		/* modem answer-chat */
	{ "al", NULL, NULL },		/* user to auto-login */
	{ "df", datefmt, NULL },	/* format for strftime() */
	{ "iM" , NULL, NULL },		/* initial message program */
	{ NULL, NULL, NULL }
};

struct	gettynums gettynums[] = {
	{ "is", 0, 0, 0 },		/* input speed */
	{ "os", 0, 0, 0 },		/* output speed */
	{ "sp", 0, 0, 0 },		/* both speeds */
	{ "nd", 0, 0, 0 },		/* newline delay */
	{ "cd", 0, 0, 0 },		/* carriage-return delay */
	{ "td", 0, 0, 0 },		/* tab delay */
	{ "fd", 0, 0, 0 },		/* form-feed delay */
	{ "bd", 0, 0, 0 },		/* backspace delay */
	{ "to", 0, 0, 0 },		/* timeout */
	{ "f0", 0, 0, 0 },		/* output flags */
	{ "f1", 0, 0, 0 },		/* input flags */
	{ "f2", 0, 0, 0 },		/* user mode flags */
	{ "pf", 0, 0, 0 },		/* delay before flush at 1st prompt */
	{ "c0", 0, 0, 0 },		/* output c_flags */
	{ "c1", 0, 0, 0 },		/* input c_flags */
	{ "c2", 0, 0, 0 },		/* user mode c_flags */
	{ "i0", 0, 0, 0 },		/* output i_flags */
	{ "i1", 0, 0, 0 },		/* input i_flags */
	{ "i2", 0, 0, 0 },		/* user mode i_flags */
	{ "l0", 0, 0, 0 },		/* output l_flags */
	{ "l1", 0, 0, 0 },		/* input l_flags */
	{ "l2", 0, 0, 0 },		/* user mode l_flags */
	{ "o0", 0, 0, 0 },		/* output o_flags */
	{ "o1", 0, 0, 0 },		/* input o_flags */
	{ "o2", 0, 0, 0 },		/* user mode o_flags */
	{ "de", 0, 0, 0 },		/* delay before sending 1st prompt */
	{ "rt", 0, 0, 0 },		/* reset timeout */
	{ "ct", 0, 0, 0 },		/* chat script timeout */
	{ "dc", 0, 0, 0 },		/* debug chat script value */
	{ NULL, 0, 0, 0 }
};


struct	gettyflags gettyflags[] = {
	{ "ht",	0, 0, 0, 0 },		/* has tabs */
	{ "nl",	1, 0, 0, 0 },		/* has newline char */
	{ "ep",	0, 0, 0, 0 },		/* even parity */
	{ "op",	0, 0, 0, 0 },		/* odd parity */
	{ "ap",	0, 0, 0, 0 },		/* any parity */
	{ "ec",	1, 0, 0, 0 },		/* no echo */
	{ "co",	0, 0, 0, 0 },		/* console special */
	{ "cb",	0, 0, 0, 0 },		/* crt backspace */
	{ "ck",	0, 0, 0, 0 },		/* crt kill */
	{ "ce",	0, 0, 0, 0 },		/* crt erase */
	{ "pe",	0, 0, 0, 0 },		/* printer erase */
	{ "rw",	1, 0, 0, 0 },		/* don't use raw */
	{ "xc",	1, 0, 0, 0 },		/* don't ^X ctl chars */
	{ "lc",	0, 0, 0, 0 },		/* terminal las lower case */
	{ "uc",	0, 0, 0, 0 },		/* terminal has no lower case */
	{ "ig",	0, 0, 0, 0 },		/* ignore garbage */
	{ "ps",	0, 0, 0, 0 },		/* do port selector speed select */
	{ "hc",	1, 0, 0, 0 },		/* don't set hangup on close */
	{ "ub", 0, 0, 0, 0 },		/* unbuffered output */
	{ "ab", 0, 0, 0, 0 },		/* auto-baud detect with '\r' */
	{ "dx", 0, 0, 0, 0 },		/* set decctlq */
	{ "np", 0, 0, 0, 0 },		/* no parity at all (8bit chars) */
	{ "mb", 0, 0, 0, 0 },		/* do MDMBUF flow control */
	{ "hw", 0, 0, 0, 0 },		/* do CTSRTS flow control */
	{ "nc", 0, 0, 0, 0 },		/* set clocal (no carrier) */
	{ "pl", 0, 0, 0, 0 },		/* use PPP instead of login(1) */
	{ NULL, 0, 0, 0, 0 }
};
