/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 27, 2024.
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
#include <string.h>

#include "sudoers.h"

static int sudoers_debug_instance = SUDO_DEBUG_INSTANCE_INITIALIZER;
static unsigned int sudoers_debug_refcnt;

static const char *const sudoers_subsystem_names[] = {
    "alias",
    "audit",
    "auth",
    "defaults",
    "env",
    "event",
    "ldap",
    "logging",
    "main",
    "match",
    "netif",
    "nss",
    "parser",
    "perms",
    "plugin",
    "rbtree",
    "sssd",
    "util",
    NULL
};

#define NUM_SUBSYSTEMS  (nitems(sudoers_subsystem_names) - 1)

/* Subsystem IDs assigned at registration time. */
unsigned int sudoers_subsystem_ids[NUM_SUBSYSTEMS];

/*
 * Parse the "filename flags,..." debug_flags entry and insert a new
 * sudo_debug_file struct into debug_files.
 */
bool
sudoers_debug_parse_flags(struct sudo_conf_debug_file_list *debug_files,
    const char *entry)
{
    /* Already initialized? */
    if (sudoers_debug_instance != SUDO_DEBUG_INSTANCE_INITIALIZER)
	return true;

    return sudo_debug_parse_flags(debug_files, entry) != -1;
}

/*
 * Register the specified debug files and program with the
 * debug subsystem, freeing the debug list when done.
 * Sets the active debug instance as a side effect.
 */
bool
sudoers_debug_register(const char *program,
    struct sudo_conf_debug_file_list *debug_files)
{
    int instance = sudoers_debug_instance;
    struct sudo_debug_file *debug_file, *debug_next;

    /* Setup debugging if indicated. */
    if (debug_files != NULL && !TAILQ_EMPTY(debug_files)) {
	if (program != NULL) {
	    instance = sudo_debug_register(program, sudoers_subsystem_names,
		sudoers_subsystem_ids, debug_files, -1);
	}
	TAILQ_FOREACH_SAFE(debug_file, debug_files, entries, debug_next) {
	    TAILQ_REMOVE(debug_files, debug_file, entries);
	    free(debug_file->debug_file);
	    free(debug_file->debug_flags);
	    free(debug_file);
	}
    }

    switch (instance) {
    case SUDO_DEBUG_INSTANCE_ERROR:
	return false;
    case SUDO_DEBUG_INSTANCE_INITIALIZER:
	/* Nothing to do */
	break;
    default:
	/* New debug instance or additional reference on existing one. */
	sudoers_debug_instance = instance;
	sudo_debug_set_active_instance(sudoers_debug_instance);
	sudoers_debug_refcnt++;
	break;
    }

    return true;
}

/*
 * Deregister sudoers_debug_instance if it is registered.
 */
void
sudoers_debug_deregister(void)
{
    debug_decl(sudoers_debug_deregister, SUDOERS_DEBUG_PLUGIN);

    if (sudoers_debug_refcnt != 0) {
	sudo_debug_exit(__func__, __FILE__, __LINE__, sudo_debug_subsys);
	if (--sudoers_debug_refcnt == 0) {
	    if (sudo_debug_deregister(sudoers_debug_instance) < 1)
		sudoers_debug_instance = SUDO_DEBUG_INSTANCE_INITIALIZER;
	}
    }
}
