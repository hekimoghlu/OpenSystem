/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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
 * AT&T Research
 *
 * output printf prompt and read response
 * if format==0 then verify that interaction is possible
 *
 * return:
 *
 *	0	[1yY+]
 *	-1	[qQ] or EOF
 *	1	otherwise
 *
 * if (quit&ERROR_PROMPT) then tty forced for IO
 * if quit>=0 then [qQ] or EOF calls exit(quit)
 */

#include <ast.h>
#include <error.h>

int
astquery(int quit, const char* format, ...)
{
	va_list		ap;
	register int	n;
	register int	c;
	int		r;
	Sfio_t*		ip;
	Sfio_t*		op;

	static Sfio_t*	rfp;
	static Sfio_t*	wfp;

	r = 0;
	va_start(ap, format);
	if (!format)
		goto done;
	r = -1;
	if (!rfp)
	{
		c = errno;
		if (isatty(sffileno(sfstdin)))
			rfp = sfstdin;
		else if (!(rfp = sfopen(NiL, "/dev/tty", "r")))
			goto done;
		if (isatty(sffileno(sfstderr)))
			wfp = sfstderr;
		else if (!(wfp = sfopen(NiL, "/dev/tty", "w")))
			goto done;
		errno = c;
	}
	if (quit & ERROR_PROMPT)
	{
		quit &= ~ERROR_PROMPT;
		ip = rfp;
		op = wfp;
	}
	else
	{
		ip = sfstdin;
		op = sfstderr;
	}
	sfsync(sfstdout);
	sfvprintf(op, format, ap);
	sfsync(op);
	for (n = c = sfgetc(ip);; c = sfgetc(ip))
		switch (c)
		{
		case EOF:
			n = c;
			/*FALLTHROUGH*/
		case '\n':
			switch (n)
			{
			case EOF:
			case 'q':
			case 'Q':
				if (quit >= 0)
					exit(quit);
				goto done;
			case '1':
			case 'y':
			case 'Y':
			case '+':
				r = 0;
				goto done;
			}
			return 1;
		}
 done:
	va_end(ap);
	return r;
}
