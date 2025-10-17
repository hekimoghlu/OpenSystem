/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
#ifndef _GLOBALS_H
#define _GLOBALS_H

#include "mechanisms.h"


/* saslauthd-main.c */
extern int              g_argc;
extern char             **g_argv;
extern int              flags;
extern int              num_procs;
extern char             *mech_option;
extern char             *run_path;
extern authmech_t       *auth_mech;


/* flags bits */
#define VERBOSE                 (1 << 1)
#define LOG_USE_SYSLOG          (1 << 2)
#define LOG_USE_STDERR          (1 << 3)
#define AM_MASTER               (1 << 4)
#define USE_ACCEPT_LOCK         (1 << 5)
#define DETACH_TTY              (1 << 6)
#define CACHE_ENABLED           (1 << 7)
#define USE_PROCESS_MODEL       (1 << 8)
#define CONCAT_LOGIN_REALM      (1 << 9)


#endif  /* _GLOBALS_H */
