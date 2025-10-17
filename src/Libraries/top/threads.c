/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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
#include "threads.h"
#include "generic.h"
#include "libtop.h"
#include "preferences.h"
#include "uinteger.h"
#include <inttypes.h>

/* Return true if an error occurred. */
static bool
threadcount_insert_cell(struct statistic *s, const void *sample)
{
	const libtop_psamp_t *psamp = sample;
	char thbuf[GENERIC_INT_SIZE];
	char runbuf[GENERIC_INT_SIZE];
	char buf[GENERIC_INT_SIZE * 2 + 1];
	struct top_uinteger thtotal, thrun;

	thtotal = top_uinteger_calc_result(psamp->th, psamp->p_th, 0ULL);
	thrun = top_uinteger_calc_result(psamp->running_th, psamp->p_running_th, 0ULL);

	if (top_sprint_uinteger(thbuf, sizeof(thbuf), thtotal))
		return true;

	if (top_sprint_uinteger(runbuf, sizeof(runbuf), thrun))
		return true;

	if (-1
			== (thrun.value > 0 ? snprintf(buf, sizeof(buf), "%s/%s", thbuf, runbuf)
								: snprintf(buf, sizeof(buf), "%s", thbuf)))
		return true;

	return generic_insert_cell(s, buf);
}

static struct statistic_callbacks callbacks = { .draw = generic_draw,
	.resize_cells = generic_resize_cells,
	.move_cells = generic_move_cells,
	.get_request_size = generic_get_request_size,
	.get_minimum_size = generic_get_minimum_size,
	.insert_cell = threadcount_insert_cell,
	.reset_insertion = generic_reset_insertion };

struct statistic *
top_threadcount_create(WINDOW *parent, const char *name)
{
	return create_statistic(STATISTIC_THREADS, parent, NULL, &callbacks, name);
}
