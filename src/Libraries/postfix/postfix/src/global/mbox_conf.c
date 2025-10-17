/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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

/* Utility library. */

#include <name_mask.h>
#include <argv.h>

/* Global library. */

#include <mail_params.h>
#include <mbox_conf.h>

 /*
  * The table with available mailbox locking methods. Some systems have
  * flock() locks; all POSIX-compatible systems have fcntl() locks. Even
  * though some systems do not use dotlock files by default (4.4BSD), such
  * locks can be necessary when accessing mailbox files over NFS.
  */
static const NAME_MASK mbox_mask[] = {
#ifdef HAS_FLOCK_LOCK
    "flock", MBOX_FLOCK_LOCK,
#endif
#ifdef HAS_FCNTL_LOCK
    "fcntl", MBOX_FCNTL_LOCK,
#endif
    "dotlock", MBOX_DOT_LOCK,
    0,
};

/* mbox_lock_mask - translate mailbox lock names to bit mask */

int     mbox_lock_mask(const char *string)
{
    return (name_mask(VAR_MAILBOX_LOCK, mbox_mask, string));
}

/* mbox_lock_names - return available mailbox lock method names */

ARGV   *mbox_lock_names(void)
{
    const NAME_MASK *np;
    ARGV   *argv;

    argv = argv_alloc(2);
    for (np = mbox_mask; np->name != 0; np++)
	argv_add(argv, np->name, ARGV_END);
    argv_terminate(argv);
    return (argv);
}
