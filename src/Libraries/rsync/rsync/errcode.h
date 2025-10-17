/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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
/* If you change these, please also update the string mappings in log.c and
 * the EXIT VALUES in rsync.yo. */

#define RERR_OK         0
#define RERR_SYNTAX     1       /* syntax or usage error */
#define RERR_PROTOCOL   2       /* protocol incompatibility */
#define RERR_FILESELECT 3       /* errors selecting input/output files, dirs */
#define RERR_UNSUPPORTED 4      /* requested action not supported */
#define RERR_STARTCLIENT 5      /* error starting client-server protocol */

#define RERR_SOCKETIO   10      /* error in socket IO */
#define RERR_FILEIO     11      /* error in file IO */
#define RERR_STREAMIO   12      /* error in rsync protocol data stream */
#define RERR_MESSAGEIO  13      /* errors with program diagnostics */
#define RERR_IPC        14      /* error in IPC code */
#define RERR_CRASHED    15      /* sibling crashed */
#define RERR_TERMINATED 16      /* sibling terminated abnormally */

#define RERR_SIGNAL1    19      /* status returned when sent SIGUSR1 */
#define RERR_SIGNAL     20      /* status returned when sent SIGINT, SIGTERM, SIGHUP */
#define RERR_WAITCHILD  21      /* some error returned by waitpid() */
#define RERR_MALLOC     22      /* error allocating core memory buffers */
#define RERR_PARTIAL    23      /* partial transfer */
#define RERR_VANISHED   24      /* file(s) vanished on sender side */
#define RERR_DEL_LIMIT  25      /* skipped some deletes due to --max-delete */

#define RERR_TIMEOUT    30      /* timeout in data send/receive */

/* Although it doesn't seem to be specified anywhere,
 * ssh and the shell seem to return these values:
 *
 * 124 if the command exited with status 255
 * 125 if the command is killed by a signal
 * 126 if the command cannot be run
 * 127 if the command is not found
 *
 * and we could use this to give a better explanation if the remote
 * command is not found.
 */
#define RERR_CMD_FAILED 124
#define RERR_CMD_KILLED 125
#define RERR_CMD_RUN 126
#define RERR_CMD_NOTFOUND 127
