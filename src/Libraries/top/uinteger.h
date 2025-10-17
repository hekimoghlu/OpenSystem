/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 8, 2023.
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

#ifndef UINTEGER_H
#define UINTEGER_H

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>

struct top_uinteger {
	bool is_negative;
	uint64_t value;
};

/* this indicates whether the sample being collected is the first
 * so that no +/- is shown next to statistics when the first sample
 * is being displayed
 */
extern bool first_sample;

struct top_uinteger top_init_uinteger(uint64_t value, bool is_negative);

struct top_uinteger top_sub_uinteger(const struct top_uinteger *a, const struct top_uinteger *b);

bool top_humanize_uinteger(char *buf, size_t bufsize, const struct top_uinteger i);

bool top_sprint_uinteger(char *buf, size_t bufsize, struct top_uinteger i);

struct top_uinteger top_uinteger_calc_result(uint64_t now, uint64_t prev, uint64_t beg);

/*
 * These return true in the case of a buffer overflow.
 * If the value has changed since the previous sample,
 * they will display a + or - to the right of the sample.
 */
bool top_uinteger_format_result(
		char *buf, size_t bufsize, uint64_t now, uint64_t prev, uint64_t beg);

bool top_uinteger_format_mem_result(
		char *buf, size_t bufsize, uint64_t now, uint64_t prev, uint64_t beg);

#endif /*UINTEGER_H*/
