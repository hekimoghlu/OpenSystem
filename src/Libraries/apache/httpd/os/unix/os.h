/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
/**
 * @file unix/os.h
 * @brief This file in included in all Apache source code. It contains definitions
 * of facilities available on _this_ operating system (HAVE_* macros),
 * and prototypes of OS specific functions defined in os.c or os-inline.c
 *
 * @defgroup APACHE_OS_UNIX unix
 * @ingroup  APACHE_OS
 * @{
 */

#ifndef APACHE_OS_H
#define APACHE_OS_H

#include "apr.h"
#include "ap_config.h"

#ifndef PLATFORM
#define PLATFORM "Unix"
#endif

/* On platforms where AP_NEED_SET_MUTEX_PERMS is defined, modules
 * should call unixd_set_*_mutex_perms on mutexes created in the
 * parent process. */
#define AP_NEED_SET_MUTEX_PERMS 1

/* Define command-line rewriting for this platform, handled by core.
 */
#define AP_PLATFORM_REWRITE_ARGS_HOOK ap_mpm_rewrite_args

#ifdef _OSD_POSIX
pid_t os_fork(const char *user);
#endif

#endif  /* !APACHE_OS_H */
/** @} */
