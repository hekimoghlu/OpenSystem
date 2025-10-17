/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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
#ifndef _PORT_TUN_H
#define _PORT_TUN_H

struct Channel;
struct ssh;

#if defined(SSH_TUN_LINUX) || defined(SSH_TUN_FREEBSD)
# define CUSTOM_SYS_TUN_OPEN
int	  sys_tun_open(int, int, char **);
#endif

#if defined(SSH_TUN_COMPAT_AF) || defined(SSH_TUN_PREPEND_AF)
# define SSH_TUN_FILTER
int	 sys_tun_infilter(struct ssh *, struct Channel *, char *, int);
u_char	*sys_tun_outfilter(struct ssh *, struct Channel *, u_char **, size_t *);
#endif

#if defined(SYS_RDOMAIN_LINUX)
# define HAVE_SYS_GET_RDOMAIN
# define HAVE_SYS_SET_RDOMAIN
# define HAVE_SYS_VALID_RDOMAIN
char *sys_get_rdomain(int fd);
int sys_set_rdomain(int fd, const char *name);
int sys_valid_rdomain(const char *name);
#endif

#if defined(SYS_RDOMAIN_XXX)
# define HAVE_SYS_SET_PROCESS_RDOMAIN
void sys_set_process_rdomain(const char *name);
#endif

#endif
