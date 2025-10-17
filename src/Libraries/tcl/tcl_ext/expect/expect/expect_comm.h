/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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
#ifndef _EXPECT_COMM_H
#define _EXPECT_COMM_H

/* common return codes for Expect functions */
/* The library actually only uses TIMEOUT and EOF */
#define EXP_ABEOF	-1	/* abnormal eof in Expect */
				/* when in library, this define is not used. */
				/* Instead "-1" is used literally in the */
				/* usual sense to check errors in system */
				/* calls */
#define EXP_TIMEOUT	-2
#define EXP_TCLERROR	-3
#define EXP_FULLBUFFER	-5
#define EXP_MATCH	-6
#define EXP_NOMATCH	-7
#define EXP_CANTMATCH	EXP_NOMATCH
#define EXP_CANMATCH	-8
#define EXP_DATA_NEW	-9	/* if select says there is new data */
#define EXP_DATA_OLD	-10	/* if we already read data in another cmd */
#define EXP_EOF		-11
#define EXP_RECONFIGURE	-12	/* changes to indirect spawn id lists */
				/* require us to reconfigure things */

/* in the unlikely event that a signal handler forces us to return this */
/* through expect's read() routine, we temporarily convert it to this. */
#define EXP_TCLRET	-20
#define EXP_TCLCNT	-21
#define EXP_TCLCNTTIMER	-22
#define EXP_TCLBRK	-23
#define EXP_TCLCNTEXP	-24
#define EXP_TCLRETTCL	-25

/* yet more TCL return codes */
/* Tcl does not safely provide a way to define the values of these, so */
/* use ridiculously different numbers for safety */
#define EXP_CONTINUE		-101	/* continue expect command */
					/* and restart timer */
#define EXP_CONTINUE_TIMER	-102	/* continue expect command */
					/* and continue timer */
#define EXP_TCL_RETURN		-103	/* converted by interact */
					/* and interpeter from */
					/* inter_return into */
					/* TCL_RETURN*/

/*
 * Everything below here should eventually be moved into expect.h
 * and Expect-thread-safe variables.
 */

EXTERN char *exp_pty_error;		/* place to pass a string generated */
					/* deep in the innards of the pty */
					/* code but needed by anyone */
EXTERN int exp_disconnected;		/* proc. disc'd from controlling tty */


#endif /* _EXPECT_COMM_H */
