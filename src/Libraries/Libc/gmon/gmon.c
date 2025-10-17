/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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
#if defined(PROFILE)
#error This module cannot be compiled with profiling
#endif

/*-
 * Copyright (c) 1983, 1992, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
/*
 * History
 *  2-Mar-90  Gregg Kellogg (gk) at NeXT
 *	Changed include of kern/mach.h to kern/mach_interface.h
 *
 *  1-May-90  Matthew Self (mself) at NeXT
 *	Added prototypes, and added casts to remove all warnings.
 *	Made all private data static.
 *	vm_deallocate old data defore vm_allocate'ing new data.
 *	Added new functions monoutput and monreset.
 *
 *  18-Dec-92 Development Environment Group at NeXT
 *	Added multiple profile areas, the ability to profile shlibs and the
 *	ability to profile rld loaded code.  Moved the machine dependent mcount
 *	routine out of this source file.
 *
 *  13-Dec-92 Development Environment Group at NeXT
 *	Added support for dynamic shared libraries.  Also removed the code that
 *	had been ifdef'ed out for profiling fixed shared libraries and
 *	objective-C.
 * 
 *  29-Aug-11 Vishal Patel (vishal_patel) at Apple
 *	Removed code that made calls to deprecated syscalls profil() and 
 *	add_profil(). The syscalls are not supported since 2008 and planned
 *	to be completely removed soon. Similarly the monitor apis are also
 * 	deprecated.
 *
 */

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)gmon.c	5.2 (Berkeley) 6/21/85";
#endif

/*
 * see profil(2) where this (SCALE_1_TO_1) is describe (incorrectly).
 *
 * The correct description:  scale is a fixed point value with
 * the binary point in the middle of the 32 bit value.  (Bit 16 is
 * 1, bit 15 is .5, etc.)
 *
 * Setting the scale to "1" (i.e. 0x10000), results in the kernel
 * choosing the profile bucket address 1 to 1 with the pc sampled.
 * Since buckets are shorts, if the profiling base were 0, then a pc
 * of 0 increments bucket 0, a pc of 2 increments bucket 1, and a pc
 * of 4 increments bucket 2.)  (Actually, this seems a little bogus,
 * 1 to 1 should map pc's to buckets -- that's probably what was
 * intended from the man page, but historically....
 */
#define		SCALE_1_TO_1	0x10000L

#define	MSG "No space for monitor buffer(s)\n"

#include <stdio.h>
#include <libc.h>
#include <monitor.h>
#include <sys/types.h>
#include <sys/gmon.h>
#include <sys/param.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach-o/loader.h>
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>

/*
 * These are defined in here and these declarations need to be moved to libc.h
 * where the other declarations for the monitor(3) routines are declared.
 */
extern void moninit(
    void);
extern void monaddition(
    char *lowpc,
    char *highpc);
extern void moncount(
    char *frompc,
    char *selfpc);
extern void monreset(
    void);
extern void monoutput(
    const char *filename);

void
moninit(
void)
{
	return; // Deprecated api. do nothing
}

void
monstartup(
char *lowpc,
char *highpc)
{
	return; // Deprecated api. do nothing
}

/*
 * monaddtion() is used for adding additional pc ranges to profile.  This is
 * used for profiling dyld loaded code.
 */
void
monaddition(
char *lowpc,
char *highpc)
{
	return; // Deprecated api. do nothing
}

void
monreset(
void)
{
	return; // Deprecated api. do nothing
}

void
monoutput(
const char *filename)
{
	return; // Deprecated api. do nothing
}

void
monitor(
char *lowpc,
char *highpc,
char *buf,
int bufsiz,
int nfunc) /* nfunc is not used; available for compatability only. */
{
	return; // Deprecated api. do nothing
}

/*
 * Control profiling
 *	profiling is what mcount checks to see if
 *	all the data structures are ready.
 */
void
moncontrol(
int mode)
{
	return; // Deprecated api. do nothing
}

void
moncount(
char *frompc,
char *selfpc)
{
	return; //Deprecated api. do nothing
}
