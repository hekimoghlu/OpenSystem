/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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
 * Copyright 2008 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef	_RTLD_DB_H
#define	_RTLD_DB_H

#ifdef	__cplusplus
extern "C" {
#endif

#include <sys/types.h>
#include <sys/link.h>

typedef uint64_t Lmid_t;

typedef unsigned long	psaddr_t;

struct ps_prochandle;


typedef enum {
	RD_OK,		/* generic "call" succeeded */
} rd_err_e;

/*
 * ways that the event notification can take place:
 */
typedef enum {
	RD_NOTIFY_BPT,		/* set break-point at address */
} rd_notify_e;

/*
 * information on ways that the event notification can take place:
 */
typedef struct rd_notify {
	rd_notify_e	type;
	union {
		psaddr_t	bptaddr;	/* break point address */
		long		syscallno;	/* system call id */
	} u;
} rd_notify_t;

/*
 * information about event instance:
 */
typedef enum {
	RD_NOSTATE = 0,		/* no state information */
	RD_CONSISTENT,		/* link-maps are stable */
	RD_ADD,			/* currently adding object to link-maps */
} rd_state_e;

typedef struct rd_event_msg {
	rd_event_e	type;
	union {
		rd_state_e	state;	/* for DLACTIVITY */
	} u;
} rd_event_msg_t;

typedef struct rd_agent rd_agent_t;

extern char		*rd_errstr(rd_err_e rderr);
extern rd_err_e		rd_event_addr(rd_agent_t *, rd_event_e, rd_notify_t *);
extern rd_err_e		rd_event_enable(rd_agent_t *, int);
extern rd_err_e		rd_event_getmsg(rd_agent_t *, rd_event_msg_t *);

#ifdef	__cplusplus
}
#endif

#endif	/* _RTLD_DB_H */
