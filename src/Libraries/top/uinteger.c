/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
#include "uinteger.h"
#include "preferences.h"
#include <inttypes.h>
#include <libutil.h>
#include <stdbool.h>
#include <string.h>

bool first_sample = true;

struct top_uinteger
top_init_uinteger(uint64_t value, bool is_negative)
{
	struct top_uinteger r;

	r.is_negative = is_negative;
	r.value = value;

	return r;
}

struct top_uinteger
top_sub_uinteger(const struct top_uinteger *a, const struct top_uinteger *b)
{
	struct top_uinteger r;

	if (!a->is_negative && !b->is_negative) {
		if (a->value > b->value) {
			/* The value will fit without underflow. */
			r.is_negative = false;
			r.value = a->value - b->value;
		} else {
			/* B is larger or we have a r.value of 0. */
			r.is_negative = true;
			r.value = b->value - a->value;
		}
	} else if (a->is_negative && !b->is_negative) {
		/* A is negative and B is positive. */
		r.is_negative = true;
		/*
		 * The A value is negative, so actually add the amount we would subtract.
		 * Thus if a is -5 and b is 2: -5 - 2 = -7;
		 */
		r.value = a->value + b->value;
	} else if (!a->is_negative && b->is_negative) {
		/* A is positive and b is negative. */
		r.is_negative = false;
		/*
		 * If say A is 2 and b is -3 we want value to be: 2 - -3 = 5;
		 */
		r.value = a->value + b->value;
	} else {
		/* They are both negative. */
		r.is_negative = true;
		r.value = a->value + b->value;
	}

	if (0 == r.value)
		r.is_negative = 0;

	return r;
}

/* Return true if an error occurred. */
bool
top_humanize_uinteger(char *buf, size_t bufsize, struct top_uinteger i)
{

	if (i.is_negative) {
		if (-1
				== humanize_number(buf + 1, bufsize - 1, (int64_t)i.value, "", HN_AUTOSCALE,
						HN_NOSPACE | HN_B)) {
			return true;
		}
		buf[0] = '-';
	} else {
		if (-1
				== humanize_number(
						buf, bufsize, (int64_t)i.value, "", HN_AUTOSCALE, HN_NOSPACE | HN_B)) {
			return true;
		}
	}

	return false;
}

bool
top_sprint_uinteger(char *buf, size_t bufsize, struct top_uinteger i)
{

	if (i.is_negative) {
		if (-1 == snprintf(buf, bufsize, "-%" PRIu64, i.value))
			return true;
	} else {
		if (-1 == snprintf(buf, bufsize, "%" PRIu64, i.value))
			return true;
	}

	return false;
}

struct top_uinteger
top_uinteger_calc_result(uint64_t now, uint64_t prev, uint64_t beg)
{
	struct top_uinteger result, prevu, begu;

	result = top_init_uinteger(now, false);

	switch (top_prefs_get_mode()) {
	case STATMODE_ACCUM:
		begu = top_init_uinteger(beg, false);
		result = top_sub_uinteger(&result, &begu);
		break;

	case STATMODE_DELTA:
		prevu = top_init_uinteger(prev, false);
		result = top_sub_uinteger(&result, &prevu);
		break;
	}

	return result;
}

/* Return true if an error occurred. */
bool
top_uinteger_format_result(char *buf, size_t bufsize, uint64_t now, uint64_t prev, uint64_t beg)
{
	struct top_uinteger i;
	int suffix = '\0';

	i = top_uinteger_calc_result(now, prev, beg);

	if (STATMODE_DELTA == top_prefs_get_mode()) {
		/* We don't need a suffix in delta mode. */
		if (top_sprint_uinteger(buf, bufsize, i)) {
			return true;
		}
	} else {
		if (!first_sample) {
			if (now < prev) {
				/* The value has decreased since the previous sample. */
				suffix = '-';
			} else if (now > prev) {
				suffix = '+';
			}
		}

		if (-1
				== snprintf(buf, bufsize, "%s%" PRIu64 "%c", (i.is_negative ? "-" : ""), i.value,
						suffix)) {
			return true;
		}
	}

	return false;
}

/* Return true if an error occurred. */
bool
top_uinteger_format_mem_result(char *buf, size_t bufsize, uint64_t now, uint64_t prev, uint64_t beg)
{
	struct top_uinteger i;
	size_t len;

	i = top_uinteger_calc_result(now, prev, beg);

	if (STATMODE_DELTA == top_prefs_get_mode()) {
		/* We don't need a suffix in delta mode. */
		if (top_humanize_uinteger(buf, bufsize, i)) {
			return true;
		}
	} else {
		if (top_humanize_uinteger(buf, bufsize - 1, i)) {
			return true;
		}

		len = strlen(buf);

		if ((len + 2) <= bufsize && !first_sample) {
			if (now < prev) {
				buf[len] = '-';
				buf[len + 1] = '\0';
			} else if (now > prev) {
				buf[len] = '+';
				buf[len + 1] = '\0';
			}
		}
	}

	return false;
}
