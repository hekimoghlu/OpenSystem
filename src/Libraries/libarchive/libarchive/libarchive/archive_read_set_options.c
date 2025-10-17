/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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

#include "archive_read_private.h"
#include "archive_options_private.h"

static int	archive_set_format_option(struct archive *a,
		    const char *m, const char *o, const char *v);
static int	archive_set_filter_option(struct archive *a,
		    const char *m, const char *o, const char *v);
static int	archive_set_option(struct archive *a,
		    const char *m, const char *o, const char *v);

int
archive_read_set_format_option(struct archive *a, const char *m, const char *o,
    const char *v)
{
	return _archive_set_option(a, m, o, v,
	    ARCHIVE_READ_MAGIC, "archive_read_set_format_option",
	    archive_set_format_option);
}

int
archive_read_set_filter_option(struct archive *a, const char *m, const char *o,
    const char *v)
{
	return _archive_set_option(a, m, o, v,
	    ARCHIVE_READ_MAGIC, "archive_read_set_filter_option",
	    archive_set_filter_option);
}

int
archive_read_set_option(struct archive *a, const char *m, const char *o,
    const char *v)
{
	return _archive_set_option(a, m, o, v,
	    ARCHIVE_READ_MAGIC, "archive_read_set_option",
	    archive_set_option);
}

int
archive_read_set_options(struct archive *a, const char *options)
{
	return _archive_set_options(a, options,
	    ARCHIVE_READ_MAGIC, "archive_read_set_options",
	    archive_set_option);
}

static int
archive_set_format_option(struct archive *_a, const char *m, const char *o,
    const char *v)
{
	struct archive_read *a = (struct archive_read *)_a;
	size_t i;
	int r, rv = ARCHIVE_WARN, matched_modules = 0;

	for (i = 0; i < sizeof(a->formats)/sizeof(a->formats[0]); i++) {
		struct archive_format_descriptor *format = &a->formats[i];

		if (format->options == NULL || format->name == NULL)
			/* This format does not support option. */
			continue;
		if (m != NULL) {
			if (strcmp(format->name, m) != 0)
				continue;
			++matched_modules;
		}

		a->format = format;
		r = format->options(a, o, v);
		a->format = NULL;

		if (r == ARCHIVE_FATAL)
			return (ARCHIVE_FATAL);

		if (r == ARCHIVE_OK)
			rv = ARCHIVE_OK;
	}
	/* If the format name didn't match, return a special code for
	 * _archive_set_option[s]. */
	if (m != NULL && matched_modules == 0)
		return ARCHIVE_WARN - 1;
	return (rv);
}

static int
archive_set_filter_option(struct archive *_a, const char *m, const char *o,
    const char *v)
{
	(void)_a; /* UNUSED */
	(void)o; /* UNUSED */
	(void)v; /* UNUSED */

	/* If the filter name didn't match, return a special code for
	 * _archive_set_option[s]. */
	if (m != NULL)
		return ARCHIVE_WARN - 1;
	return ARCHIVE_WARN;
}

static int
archive_set_option(struct archive *a, const char *m, const char *o,
    const char *v)
{
	return _archive_set_either_option(a, m, o, v,
	    archive_set_format_option,
	    archive_set_filter_option);
}
