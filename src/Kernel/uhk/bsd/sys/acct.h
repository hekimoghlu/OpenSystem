/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
/* Copyright (c) 1995 NeXT Computer, Inc. All Rights Reserved */
/*-
 * Copyright (c) 1990, 1993, 1994
 *	The Regents of the University of California.  All rights reserved.
 * (c) UNIX System Laboratories, Inc.
 * All or some portions of this file are derived from material licensed
 * to the University of California by American Telephone and Telegraph
 * Co. or Unix System Laboratories, Inc. and are reproduced herein with
 * the permission of UNIX System Laboratories, Inc.
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
 *
 *	@(#)acct.h	8.4 (Berkeley) 1/9/95
 */
#ifndef _SYS_ACCT_H_
#define _SYS_ACCT_H_

#include <sys/appleapiopts.h>
#include <sys/cdefs.h>
#include <sys/_types/_u_int8_t.h>  /* u_int8_t */
#include <sys/_types/_u_int16_t.h> /* u_int16_t */
#include <sys/_types/_u_int32_t.h> /* u_int32_t */
#include <sys/_types/_uid_t.h>     /* uid_t */
#include <sys/_types/_gid_t.h>     /* gid_t */
#include <sys/_types/_dev_t.h>     /* dev_t */

/*
 * Accounting structures; these use a comp_t type which is a 3 bits base 8
 * exponent, 13 bit fraction ``floating point'' number.  Units are 1/AHZ
 * seconds.
 */
typedef u_int16_t comp_t;

struct acct {
	char      ac_comm[10];  /* command name */
	comp_t    ac_utime;     /* user time */
	comp_t    ac_stime;     /* system time */
	comp_t    ac_etime;     /* elapsed time */
	u_int32_t ac_btime;     /* starting time */
	uid_t     ac_uid;       /* user id */
	gid_t     ac_gid;       /* group id */
	u_int16_t ac_mem;       /* average memory usage */
	comp_t    ac_io;        /* count of IO blocks */
	dev_t     ac_tty;       /* controlling tty */

#define AFORK   0x01            /* fork'd but not exec'd */
#define ASU     0x02            /* used super-user permissions */
#define ACOMPAT 0x04            /* used compatibility mode */
#define ACORE   0x08            /* dumped core */
#define AXSIG   0x10            /* killed by a signal */
	u_int8_t  ac_flag;      /* accounting flags */
};

/*
 * 1/AHZ is the granularity of the data encoded in the comp_t fields.
 * This is not necessarily equal to hz.
 */
#define AHZ     64

#ifdef XNU_KERNEL_PRIVATE
#ifdef __APPLE_API_PRIVATE
extern struct vnode     *acctp;

__BEGIN_DECLS
int     acct_process(struct proc *p);
__END_DECLS

#endif /* __APPLE_API_PRIVATE */
#endif /* XNU_KERNEL_PRIVATE */

#endif /* ! _SYS_ACCT_H_ */
