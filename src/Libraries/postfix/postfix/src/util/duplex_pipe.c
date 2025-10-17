/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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
/* System libraries */

#include <sys_defs.h>
#include <sys/socket.h>
#include <unistd.h>

/* Utility library. */

#include "iostuff.h"
#include "sane_socketpair.h"

/* duplex_pipe - give me a duplex pipe or bust */

int     duplex_pipe(int *fds)
{
#ifdef HAS_DUPLEX_PIPE
    return (pipe(fds));
#else
    return (sane_socketpair(AF_UNIX, SOCK_STREAM, 0, fds));
#endif
}

