/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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
#include "tclInt.h"

/*
 * The panicProc variable contains a pointer to an application specific panic
 * procedure.
 */

static Tcl_PanicProc *panicProc = NULL;

/*
 * The platformPanicProc variable contains a pointer to a platform specific
 * panic procedure, if any. (TclpPanic may be NULL via a macro.)
 */

static Tcl_PanicProc *CONST platformPanicProc = TclpPanic;

/*
 *----------------------------------------------------------------------
 *
 * Tcl_SetPanicProc --
 *
 *	Replace the default panic behavior with the specified function.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Sets the panicProc variable.
 *
 *----------------------------------------------------------------------
 */

void
Tcl_SetPanicProc(
    Tcl_PanicProc *proc)
{
    panicProc = proc;
}

/*
 *----------------------------------------------------------------------
 *
 * Tcl_PanicVA --
 *
 *	Print an error message and kill the process.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The process dies, entering the debugger if possible.
 *
 *----------------------------------------------------------------------
 */

void
Tcl_PanicVA(
    CONST char *format,		/* Format string, suitable for passing to
				 * fprintf. */
    va_list argList)		/* Variable argument list. */
{
    char *arg1, *arg2, *arg3, *arg4;	/* Additional arguments (variable in
					 * number) to pass to fprintf. */
    char *arg5, *arg6, *arg7, *arg8;

    arg1 = va_arg(argList, char *);
    arg2 = va_arg(argList, char *);
    arg3 = va_arg(argList, char *);
    arg4 = va_arg(argList, char *);
    arg5 = va_arg(argList, char *);
    arg6 = va_arg(argList, char *);
    arg7 = va_arg(argList, char *);
    arg8 = va_arg(argList, char *);

    if (panicProc != NULL) {
	(void) (*panicProc)(format, arg1, arg2, arg3, arg4,
		arg5, arg6, arg7, arg8);
    } else if (platformPanicProc != NULL) {
	(void) (*platformPanicProc)(format, arg1, arg2, arg3, arg4,
		arg5, arg6, arg7, arg8);
    } else {
	(void) fprintf(stderr, format, arg1, arg2, arg3, arg4, arg5, arg6,
		arg7, arg8);
	(void) fprintf(stderr, "\n");
	(void) fflush(stderr);
	abort();
    }
}

/*
 *----------------------------------------------------------------------
 *
 * Tcl_Panic --
 *
 *	Print an error message and kill the process.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The process dies, entering the debugger if possible.
 *
 *----------------------------------------------------------------------
 */

	/* ARGSUSED */
void
Tcl_Panic(
    CONST char *format,
    ...)
{
    va_list argList;

    va_start(argList, format);
    Tcl_PanicVA(format, argList);
    va_end (argList);
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
