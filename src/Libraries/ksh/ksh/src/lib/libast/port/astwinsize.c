/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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
 * return terminal rows and cols
 */

#include <ast.h>
#include <ast_tty.h>
#include <sys/ioctl.h>

#if defined(__STDPP__directive) && defined(__STDPP__hide)
__STDPP__directive pragma pp:hide ioctl sleep
#else
#define ioctl		______ioctl
#define sleep		______sleep
#endif

#if _sys_ioctl
#include <sys/ioctl.h>
#endif

#if defined(TIOCGWINSZ)
#if _sys_stream && _sys_ptem
#include <sys/stream.h>
#include <sys/ptem.h>
#endif
#else
#if !defined(TIOCGSIZE) && !defined(TIOCGWINSZ)
#if _hdr_jioctl
#define jwinsize	winsize
#include <jioctl.h>
#else
#if _sys_jioctl
#define jwinsize	winsize
#include <sys/jioctl.h>
#endif
#endif
#endif
#endif

#if defined(__STDPP__directive) && defined(__STDPP__hide)
__STDPP__directive pragma pp:nohide ioctl sleep
#else
#undef	ioctl
#undef	sleep
#endif

static int		ttctl(int, int, void*);

void
astwinsize(int fd, register int* rows, register int* cols)
{
#ifdef	TIOCGWINSZ
#define NEED_ttctl
	struct winsize	ws;

	if (!ttctl(fd, TIOCGWINSZ, &ws) && ws.ws_col > 0 && ws.ws_row > 0)
	{
		if (rows) *rows = ws.ws_row;
		if (cols) *cols = ws.ws_col;
	}
	else
#else
#ifdef	TIOCGSIZE
#define NEED_ttctl
	struct ttysize	ts;

	if (!ttctl(fd, TIOCGSIZE, &ts) && ts.ts_lines > 0 && ts.ts_cols > 0)
	{
		if (rows) *rows = ts.ts_lines;
		if (cols) *cols = ts.ts_cols;
	}
	else
#else
#ifdef	JWINSIZE
#define NEED_ttctl
	struct winsize	ws;

	if (!ttctl(fd, JWINSIZE, &ws) && ws.bytesx > 0 && ws.bytesy > 0)
	{
		if (rows) *rows = ws.bytesy;
		if (cols) *cols = ws.bytesx;
	}
	else
#endif
#endif
#endif
	{
		char*		s;

		if (rows) *rows = (s = getenv("LINES")) ? strtol(s, NiL, 0) : 0;
		if (cols) *cols = (s = getenv("COLUMNS")) ? strtol(s, NiL, 0) : 0;
	}
}

#ifdef	NEED_ttctl

/*
 * tty ioctl() -- no cache
 */

static int
ttctl(register int fd, int op, void* tt)
{
	register int	v;

	if (fd < 0)
	{
		for (fd = 0; fd <= 2; fd++)
			if (!ioctl(fd, op, tt)) return(0);
		if ((fd = open("/dev/tty", O_RDONLY|O_cloexec)) >= 0)
		{
			v = ioctl(fd, op, tt);
			close(fd);
			return(v);
		}
	}
	else if (!ioctl(fd, op, tt)) return(0);
	return(-1);
}

#endif
