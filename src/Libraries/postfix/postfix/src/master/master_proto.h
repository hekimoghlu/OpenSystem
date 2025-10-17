/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
#define MASTER_XPORT_NAME_UNIX	"unix"	/* local IPC */
#define MASTER_XPORT_NAME_FIFO	"fifo"	/* local IPC */
#define MASTER_XPORT_NAME_INET	"inet"	/* non-local IPC */
#define MASTER_XPORT_NAME_PASS	"pass"	/* local IPC */

 /*
  * Format of a status message sent by a child process to the process
  * manager. Since this is between processes on the same machine we need not
  * worry about byte order and word length.
  */
typedef struct MASTER_STATUS {
    int     pid;			/* process ID */
    unsigned gen;			/* child generation number */
    int     avail;			/* availability */
} MASTER_STATUS;

#define MASTER_GEN_NAME	"GENERATION"	/* passed via environment */

#define MASTER_STAT_TAKEN	0	/* this one is occupied */
#define MASTER_STAT_AVAIL	1	/* this process is idle */

extern int master_notify(int, unsigned, int);	/* encapsulate status msg */

 /*
  * File descriptors inherited from the master process. The flow control pipe
  * is read by receive processes and is written to by send processes. If
  * receive processes get too far ahead they will pause for a brief moment.
  */
#define MASTER_FLOW_READ	3
#define MASTER_FLOW_WRITE	4

 /*
  * File descriptors inherited from the master process. All processes that
  * provide a given service share the same status file descriptor, and listen
  * on the same service socket(s). The kernel decides what process gets the
  * next connection. Usually the number of listening processes is small, so
  * one connection will not cause a "thundering herd" effect. When no process
  * listens on a given socket, the master process will. MASTER_LISTEN_FD is
  * actually the lowest-numbered descriptor of a sequence of descriptors to
  * listen on.
  */
#define MASTER_STATUS_FD	5	/* shared channel to parent */
#define MASTER_LISTEN_FD	6	/* accept connections here */

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*--*/

