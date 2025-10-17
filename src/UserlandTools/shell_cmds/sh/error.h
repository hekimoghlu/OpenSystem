/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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
 * We enclose jmp_buf in a structure so that we can declare pointers to
 * jump locations.  The global variable handler contains the location to
 * jump to when an exception occurs, and the global variable exception
 * contains a code identifying the exception.  To implement nested
 * exception handlers, the user should save the value of handler on entry
 * to an inner scope, set handler to point to a jmploc structure for the
 * inner scope, and restore handler on exit from the scope.
 */

#include <setjmp.h>
#include <signal.h>

struct jmploc {
	jmp_buf loc;
};

extern struct jmploc *handler;
extern volatile sig_atomic_t exception;

/* exceptions */
#define EXINT 0		/* SIGINT received */
#define EXERROR 1	/* a generic error */
#define EXEXEC 2	/* command execution failed */
#define EXEXIT 3	/* call exitshell(exitstatus) */


/*
 * These macros allow the user to suspend the handling of interrupt signals
 * over a period of time.  This is similar to SIGHOLD to or sigblock, but
 * much more efficient and portable.  (But hacking the kernel is so much
 * more fun than worrying about efficiency and portability. :-))
 */

extern volatile sig_atomic_t suppressint;
extern volatile sig_atomic_t intpending;

#define INTOFF suppressint++
#define INTON { if (--suppressint == 0 && intpending) onint(); }
#define is_int_on() suppressint
#define SETINTON(s) do { suppressint = (s); if (suppressint == 0 && intpending) onint(); } while (0)
#define FORCEINTON {suppressint = 0; if (intpending) onint();}
#define SET_PENDING_INT intpending = 1
#define CLEAR_PENDING_INT intpending = 0
#define int_pending() intpending

void exraise(int) __dead2;
void onint(void) __dead2;
void warning(const char *, ...) __printflike(1, 2);
void error(const char *, ...) __printf0like(1, 2) __dead2;
void exerror(int, const char *, ...) __printf0like(2, 3) __dead2;


/*
 * BSD setjmp saves the signal mask, which violates ANSI C and takes time,
 * so we use _setjmp instead.
 */

#define setjmp(jmploc)	_setjmp(jmploc)
#define longjmp(jmploc, val)	_longjmp(jmploc, val)
