/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 14, 2024.
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
#include "pid.h"
#include "generic.h"
#include "libtop.h"
#include <stdlib.h>
#include <string.h>

static void
get_pid_suffix(const libtop_psamp_t *psamp, char *suffix, size_t length)
{
	bool proc_is_foreign = false;
	bool proc_is_64 = false;
	bool host_is_64 = false;

#if defined(__LP64__)
	host_is_64 = true;
#endif

	switch (psamp->cputype) {
	case CPU_TYPE_X86_64:
		proc_is_64 = true;
	// FALLTHROUGH
	case CPU_TYPE_X86:
#if !defined(__i386__) && !defined(__x86_64__)
		proc_is_foreign = true;
#endif
		break;
	case CPU_TYPE_POWERPC64:
		proc_is_64 = true;
	// FALLTHROUGH
	case CPU_TYPE_POWERPC:
#if !defined(__ppc__) && !defined(__ppc64__)
		proc_is_foreign = true;
#endif
		break;
	case CPU_TYPE_ARM:
#if !defined(__arm__)
		proc_is_foreign = true;
#endif
		break;
	default:
		proc_is_foreign = true;
		break;
	}

	if (proc_is_foreign) {
		strncpy(suffix, "*", length);
	} else if (host_is_64 && !proc_is_64) {
		strncpy(suffix, "-", length);
	} else {
		strncpy(suffix, " ", length);
	}
}

static bool
pid_insert_cell(struct statistic *s, const void *sample)
{
	const libtop_psamp_t *psamp = sample;
	char buf[GENERIC_INT_SIZE + 2];
	char suffix[2];
	unsigned int p;

	get_pid_suffix(psamp, suffix, sizeof(suffix));
	p = (unsigned int)psamp->pid;

	if (-1 == snprintf(buf, sizeof(buf), "%u%s", p, suffix))
		return true;

	return generic_insert_cell(s, buf);
}

static struct statistic_callbacks callbacks = { .draw = generic_draw,
	.resize_cells = generic_resize_cells,
	.move_cells = generic_move_cells,
	.get_request_size = generic_get_request_size,
	.get_minimum_size = generic_get_minimum_size,
	.insert_cell = pid_insert_cell,
	.reset_insertion = generic_reset_insertion };

struct statistic *
top_pid_create(WINDOW *parent, const char *name)
{
	return create_statistic(STATISTIC_PID, parent, NULL, &callbacks, name);
}
