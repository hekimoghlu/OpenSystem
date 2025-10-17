/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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
 * Copyright (c) 1980, 1986, 1989, 1993
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
 *	@(#)param.c	8.3 (Berkeley) 8/20/94
 */

#include <sys/param.h>
#include <sys/systm.h>
#include <sys/socket.h>
#include <sys/vnode_internal.h>
#include <sys/file_internal.h>
#include <sys/mbuf.h>
#include <sys/domain.h>
#include <sys/kernel.h>
#include <sys/quota.h>
#include <miscfs/fifofs/fifo.h>
#include <sys/shm_internal.h>
#include <sys/aio_kern.h>

struct  timezone tz = { .tz_minuteswest = 0, .tz_dsttime = 0 };

#if !defined(__x86_64__)
#define NPROC 1000          /* Account for DEFAULT_TOTAL_CORPSES_ALLOWED by making this slightly lower than we can. */
#define NPROC_PER_UID 950
#else
#define NPROC (20 + 32 * 32)
#define NPROC_PER_UID (NPROC/2)
#endif

/* NOTE: maxproc and hard_maxproc values are subject to device specific scaling in bsd_scale_setup */
#define HNPROC 2500     /* based on thread_max */
int     maxproc = NPROC;
int     maxprocperuid = NPROC_PER_UID;

#if !defined(__x86_64__)
int hard_maxproc = NPROC;       /* hardcoded limit -- for ARM the number of processes is limited by the ASID space */
#else
int hard_maxproc = HNPROC;      /* hardcoded limit */
#endif

int nprocs = 0; /* XXX */

int desiredvnodes = 0;          /* desiredvnodes is set explicitly in unix_startup.c */
uint32_t kern_maxvnodes = 0;    /* global, to be read from the device tree */

#if __LP64__
int     maxfiles = 3 * OPEN_MAX;
#else
int     maxfiles = OPEN_MAX + 2048;
#endif

unsigned int    ncallout = 16 + 2 * NPROC;
unsigned int nmbclusters = NMBCLUSTERS;
int     nport = NPROC / 2;

/*
 *  async IO (aio) configurable limits
 */
int aio_max_requests = CONFIG_AIO_MAX;
int aio_max_requests_per_process = CONFIG_AIO_PROCESS_MAX;
int aio_worker_threads = CONFIG_AIO_THREAD_COUNT;

struct  buf *buf_headers;
struct domains_head domains = TAILQ_HEAD_INITIALIZER(domains);

