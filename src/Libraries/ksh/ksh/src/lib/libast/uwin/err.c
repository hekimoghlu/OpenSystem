/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 26, 2023.
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
#include "FEATURE/uwin"

#if !_UWIN

void _STUB_err(){}

#else

#pragma prototyped

/*
 * bsd 4.4 compatibility
 *
 * NOTE: errorv(ERROR_NOID) => the first arg is the printf format
 */

#include <ast.h>
#include <error.h>

#include <windows.h>

#ifdef __EXPORT__
#define extern	__EXPORT__
#endif

static void
errmsg(int level, int code, const char* fmt, va_list ap)
{
	if (!error_info.id)
	{
		struct _astdll*	dp = _ast_getdll();
		char*		s;
		char*		t;

		if (s = dp->_ast__argv[0])
		{
			if (t = strrchr(s, '/'))
				s = t + 1;
			error_info.id = s;
		}
	}
	errorv(fmt, level|ERROR_NOID, ap);
	if ((level & ERROR_LEVEL) >= ERROR_ERROR)
		exit(code);
}

extern void verr(int code, const char* fmt, va_list ap)
{
	errmsg(ERROR_ERROR|ERROR_SYSTEM, code, fmt, ap);
}

extern void err(int code, const char* fmt, ...)
{
	va_list	ap;

	va_start(ap, fmt);
	errmsg(ERROR_ERROR|ERROR_SYSTEM, code, fmt, ap);
	va_end(ap);
}

extern void verrx(int code, const char* fmt, va_list ap)
{
	errmsg(ERROR_ERROR, code, fmt, ap);
}

extern void errx(int code, const char* fmt, ...)
{
	va_list	ap;

	va_start(ap, fmt);
	errmsg(ERROR_ERROR, code, fmt, ap);
	va_end(ap);
}

extern void vwarn(const char* fmt, va_list ap)
{
	errmsg(ERROR_WARNING|ERROR_SYSTEM, 0, fmt, ap);
}

extern void warn(const char* fmt, ...)
{
	va_list	ap;

	va_start(ap, fmt);
	errmsg(ERROR_WARNING|ERROR_SYSTEM, 0, fmt, ap);
	va_end(ap);
}

extern void vwarnx(const char* fmt, va_list ap)
{
	errmsg(ERROR_WARNING, 0, fmt, ap);
}

extern void warnx(const char* fmt, ...)
{
	va_list	ap;

	va_start(ap, fmt);
	errmsg(ERROR_WARNING, 0, fmt, ap);
	va_end(ap);
}

#endif
