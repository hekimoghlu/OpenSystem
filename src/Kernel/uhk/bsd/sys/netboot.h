/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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
 * netboot.h
 * - definitions for network booting/rooting
 */

#ifndef _SYS_NETBOOT_H
#define _SYS_NETBOOT_H

#include <mach/boolean.h>
#include <netinet/in.h>

int             netboot_setup(void);
int             netboot_mountroot(void);
int             netboot_root(void);

boolean_t       netboot_iaddr(struct in_addr * iaddr_p);

boolean_t       netboot_rootpath(struct in_addr * server_ip,
    char * name, size_t name_len,
    char * path, size_t path_len);

#endif /* _SYS_NETBOOT_H */
