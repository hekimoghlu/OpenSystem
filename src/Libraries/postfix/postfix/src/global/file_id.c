/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 8, 2024.
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
/* System library. */

#include <sys_defs.h>
#include <sys/stat.h>
#include <string.h>

/* Utility library */

#include <msg.h>
#include <vstring.h>
#include <warn_stat.h>

/* Global library. */

#define MAIL_QUEUE_INTERNAL
#include <mail_queue.h>
#include "file_id.h"

/* get_file_id - binary compatibility */

const char *get_file_id(int fd)
{
    return (get_file_id_fd(fd, 0));
}

/* get_file_id_fd - return printable file identifier for file descriptor */

const char *get_file_id_fd(int fd, int long_flag)
{
    struct stat st;

    if (fstat(fd, &st) < 0)
	msg_fatal("fstat: %m");
    return (get_file_id_st(&st, long_flag));
}

/* get_file_id_st - return printable file identifier for file status */

const char *get_file_id_st(struct stat * st, int long_flag)
{
    static VSTRING *result;

    if (result == 0)
	result = vstring_alloc(1);
    if (long_flag)
	return (MQID_LG_ENCODE_INUM(result, st->st_ino));
    else
	return (MQID_SH_ENCODE_INUM(result, st->st_ino));
}
