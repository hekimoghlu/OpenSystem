/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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
#ifndef GENERIC_H
#define GENERIC_H
#include "statistic.h"

/* A value large enough for a 64-bit number string rep (greater actually). */
enum { GENERIC_INT_SIZE = 30 };

struct generic_cell {
	char *string;
	size_t allocated_length;
	size_t length;
};

struct generic_cells {
	struct generic_cell *array;
	int max_width; /* The maximum width for a generic_cell. */
	size_t length;
	size_t length_allocated;
};

enum {
	GENERIC_DRAW_LEFT,     /* Draw with an achor to the left. */
	GENERIC_DRAW_CENTERED, /* Draw text as centered as possible. */
	GENERIC_DRAW_RIGHT     /* Draw text anchored to the right. */
};

void generic_draw_aligned(struct statistic *s, int x);
void generic_draw(struct statistic *s, int x);
void generic_draw_extended(
		struct statistic *s, int x, int y, int anchor, const char *string, int slen);
void generic_draw_centered(struct statistic *s, int x);
void generic_draw_right(struct statistic *s, int x);
bool generic_resize_cells(struct statistic *s, struct statistic_size *size);
bool generic_move_cells(struct statistic *s, int x, int y);
void generic_get_request_size(struct statistic *s);
bool generic_insert_cell(struct statistic *s, const char *sample);
void generic_reset_insertion(struct statistic *s);
void generic_get_minimum_size(struct statistic *s);

#endif /*GENERIC_H*/
