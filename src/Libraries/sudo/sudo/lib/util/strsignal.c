/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#ifndef HAVE_STRSIGNAL

#include <signal.h>

#include "sudo_compat.h"
#include "sudo_gettext.h"

#if defined(HAVE_DECL_SYS_SIGLIST) && HAVE_DECL_SYS_SIGLIST == 1
# define sudo_sys_siglist	sys_siglist
#elif defined(HAVE_DECL__SYS_SIGLIST) && HAVE_DECL__SYS_SIGLIST == 1
# define sudo_sys_siglist	_sys_siglist
#else
extern const char *const sudo_sys_siglist[NSIG];
#endif

/*
 * Get signal description string
 */
char *
sudo_strsignal(int signo)
{
    if (signo > 0 && signo < NSIG && sudo_sys_siglist[signo] != NULL)
	return (char *)sudo_sys_siglist[signo];
    /* XXX - should be "Unknown signal: %d" */
    return (char *)_("Unknown signal");
}
#endif /* HAVE_STRSIGNAL */
