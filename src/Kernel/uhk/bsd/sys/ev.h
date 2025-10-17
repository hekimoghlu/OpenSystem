/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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
/* Copyright (c) 1998 Apple Computer, Inc. All rights reserved */

#ifndef _SYS_EV_H_
#define _SYS_EV_H_

#include <sys/appleapiopts.h>

#include <sys/queue.h>
#include <sys/cdefs.h>

struct eventreq {
	int      er_type;
#define EV_FD 1    // file descriptor
	int      er_handle;
	void    *er_data;
	int      er_rcnt;
	int      er_wcnt;
	int      er_ecnt;
	int      er_eventbits;
#define EV_RE  1
#define EV_WR  2
#define EV_EX  4
#define EV_RM  8
#define EV_MASK 0xf
};

typedef struct eventreq *er_t;

#define EV_RBYTES 0x100
#define EV_WBYTES 0x200
#define EV_RWBYTES (EV_RBYTES|EV_WBYTES)
#define EV_RCLOSED 0x400
#define EV_RCONN   0x800
#define EV_WCLOSED 0x1000
#define EV_WCONN   0x2000
#define EV_OOB     0x4000
#define EV_FIN     0x8000
#define EV_RESET   0x10000
#define EV_TIMEOUT 0x20000
#define EV_DMASK   0xffffff00


#ifndef KERNEL

__BEGIN_DECLS
int     modwatch(er_t, int);
int     watchevent(er_t, int);
int     waitevent(er_t, struct timeval *);
__END_DECLS

#endif

#endif /* _SYS_EV_H_ */
