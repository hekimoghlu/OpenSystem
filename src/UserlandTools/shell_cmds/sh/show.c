/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#ifndef lint
#if 0
static char sccsid[] = "@(#)show.c	8.3 (Berkeley) 5/4/95";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: head/bin/sh/show.c 326025 2017-11-20 19:49:47Z pfg $");

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>

#include "shell.h"
#include "parser.h"
#include "nodes.h"
#include "mystring.h"
#include "show.h"


#ifdef DEBUG
static void shtree(union node *, int, char *, FILE*);
static void shcmd(union node *, FILE *);
static void sharg(union node *, FILE *);
static void indent(int, char *, FILE *);
static void trstring(char *);


void
showtree(union node *n)
{
	trputs("showtree called\n");
	shtree(n, 1, NULL, stdout);
}


static void
shtree(union node *n, int ind, char *pfx, FILE *fp)
{
	struct nodelist *lp;
	char *s;

	if (n == NULL)
		return;

	indent(ind, pfx, fp);
	switch(n->type) {
	case NSEMI:
		s = "; ";
		goto binop;
	case NAND:
		s = " && ";
		goto binop;
	case NOR:
		s = " || ";
binop:
		shtree(n->nbinary.ch1, ind, NULL, fp);
	   /*    if (ind < 0) */
			fputs(s, fp);
		shtree(n->nbinary.ch2, ind, NULL, fp);
		break;
	case NCMD:
		shcmd(n, fp);
		if (ind >= 0)
			putc('\n', fp);
		break;
	case NPIPE:
		for (lp = n->npipe.cmdlist ; lp ; lp = lp->next) {
			shcmd(lp->n, fp);
			if (lp->next)
				fputs(" | ", fp);
		}
		if (n->npipe.backgnd)
			fputs(" &", fp);
		if (ind >= 0)
			putc('\n', fp);
		break;
	default:
		fprintf(fp, "<node type %d>", n->type);
		if (ind >= 0)
			putc('\n', fp);
		break;
	}
}



static void
shcmd(union node *cmd, FILE *fp)
{
	union node *np;
	int first;
	char *s;
	int dftfd;

	first = 1;
	for (np = cmd->ncmd.args ; np ; np = np->narg.next) {
		if (! first)
			putchar(' ');
		sharg(np, fp);
		first = 0;
	}
	for (np = cmd->ncmd.redirect ; np ; np = np->nfile.next) {
		if (! first)
			putchar(' ');
		switch (np->nfile.type) {
			case NTO:	s = ">";  dftfd = 1; break;
			case NAPPEND:	s = ">>"; dftfd = 1; break;
			case NTOFD:	s = ">&"; dftfd = 1; break;
			case NCLOBBER:	s = ">|"; dftfd = 1; break;
			case NFROM:	s = "<";  dftfd = 0; break;
			case NFROMTO:	s = "<>"; dftfd = 0; break;
			case NFROMFD:	s = "<&"; dftfd = 0; break;
			case NHERE:	s = "<<"; dftfd = 0; break;
			case NXHERE:	s = "<<"; dftfd = 0; break;
			default:  	s = "*error*"; dftfd = 0; break;
		}
		if (np->nfile.fd != dftfd)
			fprintf(fp, "%d", np->nfile.fd);
		fputs(s, fp);
		if (np->nfile.type == NTOFD || np->nfile.type == NFROMFD) {
			if (np->ndup.dupfd >= 0)
				fprintf(fp, "%d", np->ndup.dupfd);
			else
				fprintf(fp, "-");
		} else if (np->nfile.type == NHERE) {
				fprintf(fp, "HERE");
		} else if (np->nfile.type == NXHERE) {
				fprintf(fp, "XHERE");
		} else {
			sharg(np->nfile.fname, fp);
		}
		first = 0;
	}
}



static void
sharg(union node *arg, FILE *fp)
{
	char *p;
	struct nodelist *bqlist;
	int subtype;

	if (arg->type != NARG) {
		printf("<node type %d>\n", arg->type);
		fflush(stdout);
		abort();
	}
	bqlist = arg->narg.backquote;
	for (p = arg->narg.text ; *p ; p++) {
		switch (*p) {
		case CTLESC:
			putc(*++p, fp);
			break;
		case CTLVAR:
			putc('$', fp);
			putc('{', fp);
			subtype = *++p;
			if (subtype == VSLENGTH)
				putc('#', fp);

			while (*p != '=')
				putc(*p++, fp);

			if (subtype & VSNUL)
				putc(':', fp);

			switch (subtype & VSTYPE) {
			case VSNORMAL:
				putc('}', fp);
				break;
			case VSMINUS:
				putc('-', fp);
				break;
			case VSPLUS:
				putc('+', fp);
				break;
			case VSQUESTION:
				putc('?', fp);
				break;
			case VSASSIGN:
				putc('=', fp);
				break;
			case VSTRIMLEFT:
				putc('#', fp);
				break;
			case VSTRIMLEFTMAX:
				putc('#', fp);
				putc('#', fp);
				break;
			case VSTRIMRIGHT:
				putc('%', fp);
				break;
			case VSTRIMRIGHTMAX:
				putc('%', fp);
				putc('%', fp);
				break;
			case VSLENGTH:
				break;
			default:
				printf("<subtype %d>", subtype);
			}
			break;
		case CTLENDVAR:
		     putc('}', fp);
		     break;
		case CTLBACKQ:
		case CTLBACKQ|CTLQUOTE:
			putc('$', fp);
			putc('(', fp);
			shtree(bqlist->n, -1, NULL, fp);
			putc(')', fp);
			break;
		default:
			putc(*p, fp);
			break;
		}
	}
}


static void
indent(int amount, char *pfx, FILE *fp)
{
	int i;

	for (i = 0 ; i < amount ; i++) {
		if (pfx && i == amount - 1)
			fputs(pfx, fp);
		putc('\t', fp);
	}
}


/*
 * Debugging stuff.
 */


FILE *tracefile;

#if DEBUG >= 2
int debug = 1;
#else
int debug = 0;
#endif


void
trputc(int c)
{
	if (tracefile == NULL)
		return;
	putc(c, tracefile);
	if (c == '\n')
		fflush(tracefile);
}


void
sh_trace(const char *fmt, ...)
{
	va_list va;
	va_start(va, fmt);
	if (tracefile != NULL) {
		(void) vfprintf(tracefile, fmt, va);
		if (strchr(fmt, '\n'))
			(void) fflush(tracefile);
	}
	va_end(va);
}


void
trputs(const char *s)
{
	if (tracefile == NULL)
		return;
	fputs(s, tracefile);
	if (strchr(s, '\n'))
		fflush(tracefile);
}


static void
trstring(char *s)
{
	char *p;
	char c;

	if (tracefile == NULL)
		return;
	putc('"', tracefile);
	for (p = s ; *p ; p++) {
		switch (*p) {
		case '\n':  c = 'n';  goto backslash;
		case '\t':  c = 't';  goto backslash;
		case '\r':  c = 'r';  goto backslash;
		case '"':  c = '"';  goto backslash;
		case '\\':  c = '\\';  goto backslash;
		case CTLESC:  c = 'e';  goto backslash;
		case CTLVAR:  c = 'v';  goto backslash;
		case CTLVAR+CTLQUOTE:  c = 'V';  goto backslash;
		case CTLBACKQ:  c = 'q';  goto backslash;
		case CTLBACKQ+CTLQUOTE:  c = 'Q';  goto backslash;
backslash:	  putc('\\', tracefile);
			putc(c, tracefile);
			break;
		default:
			if (*p >= ' ' && *p <= '~')
				putc(*p, tracefile);
			else {
				putc('\\', tracefile);
				putc(*p >> 6 & 03, tracefile);
				putc(*p >> 3 & 07, tracefile);
				putc(*p & 07, tracefile);
			}
			break;
		}
	}
	putc('"', tracefile);
}


void
trargs(char **ap)
{
	if (tracefile == NULL)
		return;
	while (*ap) {
		trstring(*ap++);
		if (*ap)
			putc(' ', tracefile);
		else
			putc('\n', tracefile);
	}
	fflush(tracefile);
}


void
opentrace(void)
{
	char s[100];
	int flags;

	if (!debug)
		return;
#ifdef not_this_way
	{
		char *p;
		if ((p = getenv("HOME")) == NULL) {
			if (geteuid() == 0)
				p = "/";
			else
				p = "/tmp";
		}
		strcpy(s, p);
		strcat(s, "/trace");
	}
#else
	strcpy(s, "./trace");
#endif /* not_this_way */
	if ((tracefile = fopen(s, "a")) == NULL) {
		fprintf(stderr, "Can't open %s: %s\n", s, strerror(errno));
		return;
	}
	if ((flags = fcntl(fileno(tracefile), F_GETFL, 0)) >= 0)
		fcntl(fileno(tracefile), F_SETFL, flags | O_APPEND);
	fputs("\nTracing started.\n", tracefile);
	fflush(tracefile);
}
#endif /* DEBUG */
