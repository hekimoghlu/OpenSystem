/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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
#include "command.h"
#include "generic.h"
#include "libtop.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static bool
command_insert_cell(struct statistic *s, const void *sample)
{
	const libtop_psamp_t *psamp = sample;

	assert(NULL != psamp->command);

	return generic_insert_cell(s, psamp->command);
}

static void
command_get_minimum_size(struct statistic *s)
{
	s->minimum_size.width = 12;
	s->minimum_size.height = 1;
}

static struct statistic_callbacks callbacks = { .draw = generic_draw,
	.resize_cells = generic_resize_cells,
	.move_cells = generic_move_cells,
	.get_request_size = generic_get_request_size,
	.get_minimum_size = command_get_minimum_size,
	.insert_cell = command_insert_cell,
	.reset_insertion = generic_reset_insertion };

struct statistic *
top_command_create(WINDOW *parent, const char *name)
{
	return create_statistic(STATISTIC_COMMAND, parent, NULL, &callbacks, name);
}
