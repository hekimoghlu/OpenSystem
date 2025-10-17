/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 1, 2021.
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
/****************************************************************************
 *  Author: Thomas E. Dickey 2001                                           *
 ****************************************************************************/

#include <curses.priv.h>

#include <ctype.h>

MODULE_ID("$Id: varargs.c,v 1.11 2012/10/27 21:03:28 tom Exp $")

#ifdef TRACE

#define MAX_PARMS 10

typedef enum {
    atUnknown = 0, atInteger, atFloat, atPoint, atString
} ARGTYPE;

#define VA_INT(type) ival = (int) va_arg(ap, type)
#define VA_FLT(type) fval = va_arg(ap, type)
#define VA_PTR(type) pval = (char *)va_arg(ap, type)
#define VA_STR(type) sval = va_arg(ap, type)

#define MyBuffer _nc_globals.tracearg_buf
#define MyLength _nc_globals.tracearg_used

/*
 * Returns a string that represents the parameter list of a printf-style call.
 */
NCURSES_EXPORT(char *)
_nc_varargs(const char *fmt, va_list ap)
{
    static char dummy[] = "";

    char buffer[BUFSIZ];
    const char *param;
    int n;

    if (fmt == 0 || *fmt == '\0')
	return dummy;
    if (MyLength == 0)
	MyBuffer = typeMalloc(char, MyLength = BUFSIZ);
    if (MyBuffer == 0)
	return dummy;
    *MyBuffer = '\0';

    while (*fmt != '\0') {
	if (*fmt == '%') {
	    char *pval = 0;	/* avoid const-cast */
	    const char *sval = "";
	    double fval = 0.0;
	    int done = FALSE;
	    int ival = 0;
	    int type = 0;
	    ARGTYPE parm[MAX_PARMS];
	    int parms = 0;
	    ARGTYPE used = atUnknown;

	    while (*++fmt != '\0' && !done) {

		if (*fmt == '*') {
		    VA_INT(int);
		    if (parms < MAX_PARMS)
			parm[parms++] = atInteger;
		} else if (isalpha(UChar(*fmt))) {
		    done = TRUE;
		    switch (*fmt) {
		    case 'Z':	/* FALLTHRU */
		    case 'h':	/* FALLTHRU */
		    case 'l':	/* FALLTHRU */
			done = FALSE;
			type = *fmt;
			break;
		    case 'i':	/* FALLTHRU */
		    case 'd':	/* FALLTHRU */
		    case 'u':	/* FALLTHRU */
		    case 'x':	/* FALLTHRU */
		    case 'X':	/* FALLTHRU */
			if (type == 'l')
			    VA_INT(long);
			else if (type == 'Z')
			    VA_INT(size_t);
			else
			    VA_INT(int);
			used = atInteger;
			break;
		    case 'f':	/* FALLTHRU */
		    case 'e':	/* FALLTHRU */
		    case 'E':	/* FALLTHRU */
		    case 'g':	/* FALLTHRU */
		    case 'G':	/* FALLTHRU */
			VA_FLT(double);
			used = atFloat;
			break;
		    case 'c':
			VA_INT(int);
			used = atInteger;
			break;
		    case 's':
			VA_STR(const char *);
			used = atString;
			break;
		    case 'p':
			VA_PTR(void *);
			used = atPoint;
			break;
		    case 'n':
			VA_PTR(int *);
			used = atPoint;
			break;
		    default:
			break;
		    }
		} else if (*fmt == '%') {
		    done = TRUE;
		}
		if (used != atUnknown && parms < MAX_PARMS) {
		    parm[parms++] = used;
		    for (n = 0; n < parms; ++n) {
			used = parm[n];
			param = buffer;
			switch (used) {
			case atInteger:
			    _nc_SPRINTF(buffer, _nc_SLIMIT(sizeof(buffer))
					"%d", ival);
			    break;
			case atFloat:
			    _nc_SPRINTF(buffer, _nc_SLIMIT(sizeof(buffer))
					"%f", fval);
			    break;
			case atPoint:
			    _nc_SPRINTF(buffer, _nc_SLIMIT(sizeof(buffer))
					"%p", pval);
			    break;
			case atString:
			    param = _nc_visbuf2(1, sval);
			    break;
			case atUnknown:
			default:
			    _nc_STRCPY(buffer, "?", sizeof(buffer));
			    break;
			}
			MyLength += strlen(param) + 2;
			MyBuffer = typeRealloc(char, MyLength, MyBuffer);
			if (MyBuffer != 0) {
			    _nc_SPRINTF(MyBuffer + strlen(MyBuffer),
					_nc_SLIMIT(MyLength - strlen(MyBuffer))
					", %s", param);
			}
		    }
		}
		used = atUnknown;
	    }
	} else {
	    fmt++;
	}
    }

    return (MyBuffer ? MyBuffer : dummy);
}
#else
EMPTY_MODULE(_nc_varargs)
#endif
