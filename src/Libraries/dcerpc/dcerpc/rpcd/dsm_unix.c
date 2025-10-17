/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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
**
**  NAME:
**
**      dsm_unix.c
**
**  FACILITY:
**
**      Data Storage Manager (DSM)
**
**  ABSTRACT:
**
**  The module contains any UNIX specific routines necessary for the DSM.
**
**
*/

#include "dsm_p.h"      /* private include file */
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

/*
 *  dsm__flush_file (fd)
 *
 *  Synchronize the memory representation of the file with the disk
 *  representation.
 *
 *  Returns 0 if successful, -1 if error occurred.
 */
public int dsm__flush_file (fd)
int fd;
{
    return (fsync(fd));
}

/*
 *  dsm__lock_file (fd)
 *
 *  Locks the given open fd such that other DSM instances will not be able
 *  to open it (to lock it).  File locking varies from system to system;
 *  currently BSD and SYS5 are supported.
 */

public void dsm__lock_file (fd, st)
int fd;
error_status_t *st;
{
    int result;

#ifdef BSD
    result = flock(fd, (LOCK_EX|LOCK_NB));
#else
    struct flock    farg;

    farg.l_type   = F_WRLCK;
    farg.l_whence = 0;
    farg.l_start  = 0;
    farg.l_len    = 0;
    farg.l_pid    = getpid();
    result = fcntl(fd, F_SETLK, &farg);
#endif

    if (result == -1) {
#ifdef EWOULDBLOCK
        if (errno == EWOULDBLOCK)
#else
        if (errno == EAGAIN) /* Darwin, !_POSIX_SOURCE */
#endif
            (*st) = dsm_err_file_busy;
        else
            (*st) = dsm_err_file_io_error;
    }
    else (*st) = status_ok;
}
