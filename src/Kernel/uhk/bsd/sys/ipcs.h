/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
 * NOTE: Internal ipcs.h header; all interfaces are private; if you want this
 * same information from your own program, popen(3) the ipcs(2) command and
 * parse its output, or your program may not work on future OS releases.
 */

#ifndef _SYS_IPCS_H_
#define _SYS_IPCS_H_

#include <sys/appleapiopts.h>
#include <sys/cdefs.h>

#define IPCS_MAGIC      0x00000001      /* Version */

/*
 * IPCS_command
 *
 * This is the IPCS command structure used for obtaining status about the
 * System V IPC mechanisms.  All other operations are based on the per
 * subsystem (shm, msg, ipc) *ctl entry point, which can be called once
 * this information is known.
 */

struct IPCS_command {
	int             ipcs_magic;     /* Magic number for struct layout */
	int             ipcs_op;        /* Operation to perform */
	int             ipcs_cursor;    /* Cursor for iteration functions */
	int             ipcs_datalen;   /* Length of ipcs_data area */
	void            *ipcs_data;     /* OP specific data */
};

#ifdef KERNEL_PRIVATE
#include <machine/types.h>

struct user_IPCS_command {
	int             ipcs_magic;     /* Magic number for struct layout */
	int             ipcs_op;        /* Operation to perform */
	int             ipcs_cursor;    /* Cursor for iteration functions */
	int             ipcs_datalen;   /* Length of ipcs_data area */
	user_addr_t     ipcs_data;      /* OP specific data */
};

struct user32_IPCS_command {
	int             ipcs_magic;     /* Magic number for struct layout */
	int             ipcs_op;        /* Operation to perform */
	int             ipcs_cursor;    /* Cursor for iteration functions */
	int             ipcs_datalen;   /* Length of ipcs_data area */
	user32_addr_t   ipcs_data;      /* OP specific data */
};

#endif /* KERNEL_PRIVATE */

/*
 * OP code values for 'ipcs_op'
 */
#define IPCS_SHM_CONF   0x00000001      /* Obtain shared memory config */
#define IPCS_SHM_ITER   0x00000002      /* Iterate shared memory info */

#define IPCS_SEM_CONF   0x00000010      /* Obtain semaphore config */
#define IPCS_SEM_ITER   0x00000020      /* Iterate semaphore info */

#define IPCS_MSG_CONF   0x00000100      /* Obtain message queue config */
#define IPCS_MSG_ITER   0x00000200      /* Iterate message queue info */

/*
 * Sysctl oid name values
 */
#define IPCS_SHM_SYSCTL "kern.sysv.ipcs.shm"
#define IPCS_SEM_SYSCTL "kern.sysv.ipcs.sem"
#define IPCS_MSG_SYSCTL "kern.sysv.ipcs.msg"


#endif  /* _SYS_IPCS_H_ */
