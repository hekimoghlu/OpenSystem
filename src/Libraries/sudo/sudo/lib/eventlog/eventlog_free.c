/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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

#include <stdio.h>
#include <stdlib.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_eventlog.h"
#include "sudo_util.h"

/*
 * Free the strings in a struct eventlog.
 */
void
eventlog_free(struct eventlog *evlog)
{
    int i;
    debug_decl(eventlog_free, SUDO_DEBUG_UTIL);

    if (evlog != NULL) {
	free(evlog->iolog_path);
	free(evlog->command);
	free(evlog->cwd);
	free(evlog->runchroot);
	free(evlog->runcwd);
	free(evlog->rungroup);
	free(evlog->runuser);
	free(evlog->signal_name);
	free(evlog->submithost);
	free(evlog->submituser);
	free(evlog->submitgroup);
	free(evlog->ttyname);
	if (evlog->argv != NULL) {
	    for (i = 0; evlog->argv[i] != NULL; i++)
		free(evlog->argv[i]);
	    free(evlog->argv);
	}
	if (evlog->envp != NULL) {
	    for (i = 0; evlog->envp[i] != NULL; i++)
		free(evlog->envp[i]);
	    free(evlog->envp);
	}
	free(evlog);
    }

    debug_return;
}
