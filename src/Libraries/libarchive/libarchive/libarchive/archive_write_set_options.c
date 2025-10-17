/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
#include "archive_platform.h"

#include "archive_write_private.h"
#include "archive_options_private.h"

static int	archive_set_format_option(struct archive *a,
		    const char *m, const char *o, const char *v);
static int	archive_set_filter_option(struct archive *a,
		    const char *m, const char *o, const char *v);
static int	archive_set_option(struct archive *a,
		    const char *m, const char *o, const char *v);

int
archive_write_set_format_option(struct archive *a, const char *m, const char *o,
    const char *v)
{
	return _archive_set_option(a, m, o, v,
	    ARCHIVE_WRITE_MAGIC, "archive_write_set_format_option",
	    archive_set_format_option);
}

int
archive_write_set_filter_option(struct archive *a, const char *m, const char *o,
    const char *v)
{
	return _archive_set_option(a, m, o, v,
	    ARCHIVE_WRITE_MAGIC, "archive_write_set_filter_option",
	    archive_set_filter_option);
}

int
archive_write_set_option(struct archive *a, const char *m, const char *o,
    const char *v)
{
	return _archive_set_option(a, m, o, v,
	    ARCHIVE_WRITE_MAGIC, "archive_write_set_option",
	    archive_set_option);
}

int
archive_write_set_options(struct archive *a, const char *options)
{
	return _archive_set_options(a, options,
	    ARCHIVE_WRITE_MAGIC, "archive_write_set_options",
	    archive_set_option);
}

static int
archive_set_format_option(struct archive *_a, const char *m, const char *o,
    const char *v)
{
	struct archive_write *a = (struct archive_write *)_a;

	if (a->format_name == NULL)
		return (m == NULL)?ARCHIVE_FAILED:ARCHIVE_WARN - 1;
	/* If the format name didn't match, return a special code for
	 * _archive_set_option[s]. */
	if (m != NULL && strcmp(m, a->format_name) != 0)
		return (ARCHIVE_WARN - 1);
	if (a->format_options == NULL)
		return (ARCHIVE_WARN);
	return a->format_options(a, o, v);
}

static int
archive_set_filter_option(struct archive *_a, const char *m, const char *o,
    const char *v)
{
	struct archive_write *a = (struct archive_write *)_a;
	struct archive_write_filter *filter;
	int r, rv = ARCHIVE_WARN;

	for (filter = a->filter_first; filter != NULL; filter = filter->next_filter) {
		if (filter->options == NULL)
			continue;
		if (m != NULL && strcmp(filter->name, m) != 0)
			continue;

		r = filter->options(filter, o, v);

		if (r == ARCHIVE_FATAL)
			return (ARCHIVE_FATAL);

		if (m != NULL)
			return (r);

		if (r == ARCHIVE_OK)
			rv = ARCHIVE_OK;
	}
	/* If the filter name didn't match, return a special code for
	 * _archive_set_option[s]. */
	if (rv == ARCHIVE_WARN && m != NULL)
		rv = ARCHIVE_WARN - 1;
	return (rv);
}

static int
archive_set_option(struct archive *a, const char *m, const char *o,
    const char *v)
{
	return _archive_set_either_option(a, m, o, v,
	    archive_set_format_option,
	    archive_set_filter_option);
}
