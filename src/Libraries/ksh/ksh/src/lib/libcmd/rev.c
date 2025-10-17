/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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
#pragma prototyped
/*
 * rev [-l] [file ...]
 *
 * reverse the characters or lines of one or more files
 *
 *   David Korn
 *   AT&T Laboratories
 *   dgk@research.att.com
 *
 */

static const char usage[] =
"[-?\n@(#)$Id: rev (AT&T Research) 2007-11-29 $\n]"
USAGE_LICENSE
"[+NAME?rev - reverse the characters or lines of one or more files]"
"[+DESCRIPTION?\brev\b copies one or more files to standard output "
	"reversing the order of characters on every line of the file "
	"or reversing the order of lines of the file if \b-l\b is specified.]"
"[+?If no \afile\a is given, or if the \afile\a is \b-\b, \brev\b "
        "copies from standard input starting at the current offset.]"
"[l:line?Reverse the lines of the file.]"

"\n"
"\n[file ...]\n"
"\n"
"[+EXIT STATUS?]{"
        "[+0?All files copied successfully.]"
        "[+>0?One or more files did not copy.]"
"}"
"[+SEE ALSO?\bcat\b(1), \btail\b(1)]"
;

#include	<cmd.h>
#include	<rev.h>

/*
 * reverse the characters within a line
 */
static int rev_char(Sfio_t *in, Sfio_t *out)
{
	register int c;
	register char *ep, *bp, *cp;
	register wchar_t *wp, *xp;
	register size_t n;
	register size_t w;
	if (mbwide())
	{
		wp = 0;
		w = 0;
		while(cp = bp = sfgetr(in,'\n',0))
		{
			ep = bp + (n=sfvalue(in)) - 1;
			if (n > w)
			{
				w = roundof(n + 1, 1024);
				if (!(wp = newof(wp, wchar_t, w, 0)))
				{
					error(ERROR_SYSTEM|2, "out of space");
					return 0;
				}
			}
			xp = wp;
			while (cp < ep)
				*xp++ = mbchar(cp);
			cp = bp;
			while (xp > wp)
				cp += mbconv(cp, *--xp);
			*cp++ = '\n';
			if (sfwrite(out, bp, cp - bp) < 0)
			{
				if (wp)
					free(wp);
				return -1;
			}
		}
		if (wp)
			free(wp);
	}
	else
		while(cp = bp = sfgetr(in,'\n',0))
		{
			ep = bp + (n=sfvalue(in)) -1;
			while(ep > bp)
			{
				c = *--ep;
				*ep = *bp;
				*bp++ = c;
			}
			if(sfwrite(out,cp,n)<0)
				return(-1);
		}
	return(0);
}

int
b_rev(int argc, register char** argv, Shbltin_t* context)
{
	register Sfio_t *fp;
	register char *cp;
	register int n, line=0;
	NOT_USED(argc);

	cmdinit(argc, argv, context, ERROR_CATALOG, 0);
	for (;;)
	{
		switch (optget(argv, usage))
		{
		case 'l':
			line=1;
			continue;
		case ':':
			error(2, "%s", opt_info.arg);
			break;
		case '?':
			error(ERROR_usage(2), "%s", opt_info.arg);
			break;
		}
		break;
	}
	argv += opt_info.index;
	if(error_info.errors)
		error(ERROR_usage(2),"%s",optusage((char*)0));
	n=0;
	if(cp = *argv)
		argv++;
	do
	{
		if(!cp || streq(cp,"-"))
			fp = sfstdin;
		else if(!(fp = sfopen((Sfio_t*)0,cp,"r")))
		{
			error(ERROR_system(0),"%s: cannot open",cp);
			n=1;
			continue;
		}
		if(line)
			line = rev_line(fp,sfstdout,sftell(fp));
		else
			line = rev_char(fp,sfstdout);
		if(fp!=sfstdin)
			sfclose(fp);
		if(line < 0)
			error(ERROR_system(1),"write failed");
	}
	while(cp= *argv++);
	return(n);
}
