/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 22, 2024.
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
 * Headers if any.
 */

#include "tclInt.h"

/*
 * Overrides for regguts.h definitions, if any.
 */

#define	FUNCPTR(name, args)	(*name)args
#define	MALLOC(n)		ckalloc(n)
#define	FREE(p)			ckfree(VS(p))
#define	REALLOC(p,n)		ckrealloc(VS(p),n)

/*
 * Do not insert extras between the "begin" and "end" lines - this chunk is
 * automatically extracted to be fitted into regex.h.
 */

/* --- begin --- */
/* Ensure certain things don't sneak in from system headers. */
#ifdef __REG_WIDE_T
#undef __REG_WIDE_T
#endif
#ifdef __REG_WIDE_COMPILE
#undef __REG_WIDE_COMPILE
#endif
#ifdef __REG_WIDE_EXEC
#undef __REG_WIDE_EXEC
#endif
#ifdef __REG_REGOFF_T
#undef __REG_REGOFF_T
#endif
#ifdef __REG_VOID_T
#undef __REG_VOID_T
#endif
#ifdef __REG_CONST
#undef __REG_CONST
#endif
#ifdef __REG_NOFRONT
#undef __REG_NOFRONT
#endif
#ifdef __REG_NOCHAR
#undef __REG_NOCHAR
#endif
/* Interface types */
#define	__REG_WIDE_T	Tcl_UniChar
#define	__REG_REGOFF_T	long	/* Not really right, but good enough... */
#define	__REG_VOID_T	void
#define	__REG_CONST	const
/* Names and declarations */
#define	__REG_WIDE_COMPILE	TclReComp
#define	__REG_WIDE_EXEC		TclReExec
#define	__REG_NOFRONT		/* Don't want regcomp() and regexec() */
#define	__REG_NOCHAR		/* Or the char versions */
#define	regfree		TclReFree
#define	regerror	TclReError
/* --- end --- */

/*
 * Internal character type and related.
 */

typedef Tcl_UniChar chr;	/* The type itself. */
typedef int pchr;		/* What it promotes to. */
typedef unsigned uchr;		/* Unsigned type that will hold a chr. */
typedef int celt;		/* Type to hold chr, or NOCELT */
#define	NOCELT (-1)		/* Celt value which is not valid chr */
#define	CHR(c) (UCHAR(c))	/* Turn char literal into chr literal */
#define	DIGITVAL(c) ((c)-'0')	/* Turn chr digit into its value */
#if TCL_UTF_MAX > 3
#define	CHRBITS	32		/* Bits in a chr; must not use sizeof */
#define	CHR_MIN	0x00000000	/* Smallest and largest chr; the value */
#define	CHR_MAX	0xffffffff	/* CHR_MAX-CHR_MIN+1 should fit in uchr */
#else
#define	CHRBITS	16		/* Bits in a chr; must not use sizeof */
#define	CHR_MIN	0x0000		/* Smallest and largest chr; the value */
#define	CHR_MAX	0xffff		/* CHR_MAX-CHR_MIN+1 should fit in uchr */
#endif

/*
 * Functions operating on chr.
 */

#define	iscalnum(x)	Tcl_UniCharIsAlnum(x)
#define	iscalpha(x)	Tcl_UniCharIsAlpha(x)
#define	iscdigit(x)	Tcl_UniCharIsDigit(x)
#define	iscspace(x)	Tcl_UniCharIsSpace(x)

/*
 * Name the external functions.
 */

#define	compile		TclReComp
#define	exec		TclReExec

/*
& Enable/disable debugging code (by whether REG_DEBUG is defined or not).
*/

#if 0				/* No debug unless requested by makefile. */
#define	REG_DEBUG	/* */
#endif

/*
 * Method of allocating a local workspace. We used a thread-specific data
 * space to store this because the regular expression engine is never
 * reentered from the same thread; it doesn't make any callbacks.
 */

#if 1
#define AllocVars(vPtr) \
    static Tcl_ThreadDataKey varsKey; \
    register struct vars *vPtr = (struct vars *) \
	    Tcl_GetThreadData(&varsKey, sizeof(struct vars))
#else
/*
 * This strategy for allocating workspace is "more proper" in some sense, but
 * quite a bit slower. Using TSD (as above) leads to code that is quite a bit
 * faster in practice (measured!)
 */
#define AllocVars(vPtr) \
    register struct vars *vPtr = (struct vars *) MALLOC(sizeof(struct vars))
#define FreeVars(vPtr) \
    FREE(vPtr)
#endif

/*
 * And pick up the standard header.
 */

#include "regex.h"
