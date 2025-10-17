/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
 * Copyright (c) 1995 NeXT Computer, Inc. All Rights Reserved
 */

#ifndef _LIBC_H
#define _LIBC_H

#include <stdio.h>
#include <unistd.h>

#ifdef	__STRICT_BSD__
#include <strings.h>
#include <varargs.h>
#else
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>
#endif

#include <sys/param.h>
#include <sys/mount.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/resource.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <mach/machine/vm_types.h>
#include <mach/boolean.h>
#include <mach/kern_return.h>

struct qelem {
        struct qelem *q_forw;
        struct qelem *q_back;
        char *q_data;
};

#include <sys/cdefs.h>

__BEGIN_DECLS
extern kern_return_t map_fd(int fd, vm_offset_t offset,
        vm_offset_t *addr, boolean_t find_space, vm_size_t numbytes);
__END_DECLS

#endif  /* _LIBC_H */
