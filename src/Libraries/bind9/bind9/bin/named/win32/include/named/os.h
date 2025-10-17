/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 2, 2022.
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
#ifndef NS_OS_H
#define NS_OS_H 1

#include <isc/types.h>

void
ns_os_init(const char *progname);

void
ns_os_daemonize(void);

void
ns_os_opendevnull(void);

void
ns_os_closedevnull(void);

void
ns_os_chroot(const char *root);

void
ns_os_inituserinfo(const char *username);

void
ns_os_changeuser(void);

void
ns_os_adjustnofile(void);

void
ns_os_minprivs(void);

FILE *
ns_os_openfile(const char *filename, int mode, isc_boolean_t switch_user);

void
ns_os_writepidfile(const char *filename, isc_boolean_t first_time);

void
ns_os_shutdown(void);

isc_result_t
ns_os_gethostname(char *buf, size_t len);

void
ns_os_shutdownmsg(char *command, isc_buffer_t *text);

void
ns_os_tzset(void);

void
ns_os_started(void);

char *
ns_os_uname(void);

#endif /* NS_OS_H */
