/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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
 * Glenn Fowler
 * AT&T Research
 *
 * return error message string given errno
 */

#include "lclib.h"

#include "FEATURE/errno"

#undef	strerror

#if !defined(sys_errlist) && !_def_errno_sys_errlist
#if _dat_sys_errlist
extern char*	sys_errlist[];
#else
#undef		_dat_sys_nerr
char*		sys_errlist[] = { 0 };
#endif
#endif

#if !defined(sys_nerr) && !_def_errno_sys_nerr
#if _dat_sys_nerr
extern int	sys_nerr;
#else
#undef		_dat_sys_nerr
int		sys_nerr = 0;
#endif
#endif

#if _lib_strerror
extern char*	strerror(int);
#endif

#if _PACKAGE_astsa

#define fmtbuf(n)	((n),tmp)

static char		tmp[32];

#endif

char*
_ast_strerror(int err)
{
	char*		msg;
	int		z;

#if _lib_strerror
	z = errno;
	msg = strerror(err);
	errno = z;
#else
	if (err > 0 && err <= sys_nerr)
		msg = (char*)sys_errlist[err];
	else
		msg = 0;
#endif
	if (msg)
	{
#if !_PACKAGE_astsa
		if (ERROR_translating())
		{
#if _lib_strerror
			static int	sys;

			if (!sys)
			{
				char*	s;
				char*	t;
				char*	p;

#if _lib_strerror
				/*
				 * stash the pending strerror() msg
				 */

				msg = strcpy(fmtbuf(strlen(msg) + 1), msg);
#endif

				/*
				 * make sure that strerror() translates
				 */

				if (!(s = strerror(1)))
					sys = -1;
				else
				{
					t = fmtbuf(z = strlen(s) + 1);
					strcpy(t, s);
					ast.locale.set |= AST_LC_internal;
					p = setlocale(LC_MESSAGES, NiL);
					setlocale(LC_MESSAGES, "C");
					sys = (s = strerror(1)) && strcmp(s, t) ? 1 : -1;
					setlocale(LC_MESSAGES, p);
					ast.locale.set &= ~AST_LC_internal;
				}
			}
			if (sys > 0)
				return msg;
#endif
			return ERROR_translate(NiL, NiL, "errlist", msg);
		}
#endif
		return msg;
	}
	msg = fmtbuf(z = 32);
	sfsprintf(msg, z, ERROR_translate(NiL, NiL, "errlist", "Error %d"), err);
	return msg;
}

#if !_lib_strerror

#if defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern char*
strerror(int err)
{
	return _ast_strerror(err);
}

#endif
