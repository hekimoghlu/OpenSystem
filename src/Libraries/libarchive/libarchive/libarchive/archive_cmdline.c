/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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

#ifdef HAVE_STRING_H
#  include <string.h>
#endif
#ifdef HAVE_STDLIB_H
#  include <stdlib.h>
#endif

#include "archive.h"
#include "archive_cmdline_private.h"
#include "archive_string.h"

static int cmdline_set_path(struct archive_cmdline *, const char *);
static int cmdline_add_arg(struct archive_cmdline *, const char *);

static ssize_t
extract_quotation(struct archive_string *as, const char *p)
{
	const char *s;

	for (s = p + 1; *s;) {
		if (*s == '\\') {
			if (s[1] != '\0') {
				archive_strappend_char(as, s[1]);
				s += 2;
			} else
				s++;
		} else if (*s == '"')
			break;
		else {
			archive_strappend_char(as, s[0]);
			s++;
		}
	}
	if (*s != '"')
		return (ARCHIVE_FAILED);/* Invalid sequence. */
	return ((ssize_t)(s + 1 - p));
}

static ssize_t
get_argument(struct archive_string *as, const char *p)
{
	const char *s = p;

	archive_string_empty(as);

	/* Skip beginning space characters. */
	while (*s != '\0' && *s == ' ')
		s++;
	/* Copy non-space characters. */
	while (*s != '\0' && *s != ' ') {
		if (*s == '\\') {
			if (s[1] != '\0') {
				archive_strappend_char(as, s[1]);
				s += 2;
			} else {
				s++;/* Ignore this character.*/
				break;
			}
		} else if (*s == '"') {
			ssize_t q = extract_quotation(as, s);
			if (q < 0)
				return (ARCHIVE_FAILED);/* Invalid sequence. */
			s += q;
		} else {
			archive_strappend_char(as, s[0]);
			s++;
		}
	}
	return ((ssize_t)(s - p));
}

/*
 * Set up command line arguments.
 * Returns ARCHIVE_OK if everything okey.
 * Returns ARCHIVE_FAILED if there is a lack of the `"' terminator or an
 * empty command line.
 * Returns ARCHIVE_FATAL if no memory.
 */
int
__archive_cmdline_parse(struct archive_cmdline *data, const char *cmd)
{
	struct archive_string as;
	const char *p;
	ssize_t al;
	int r;

	archive_string_init(&as);

	/* Get first argument as a command path. */
	al = get_argument(&as, cmd);
	if (al < 0) {
		r = ARCHIVE_FAILED;/* Invalid sequence. */
		goto exit_function;
	}
	if (archive_strlen(&as) == 0) {
		r = ARCHIVE_FAILED;/* An empty command path. */
		goto exit_function;
	}
	r = cmdline_set_path(data, as.s);
	if (r != ARCHIVE_OK)
		goto exit_function;
	p = strrchr(as.s, '/');
	if (p == NULL)
		p = as.s;
	else
		p++;
	r = cmdline_add_arg(data, p);
	if (r != ARCHIVE_OK)
		goto exit_function;
	cmd += al;

	for (;;) {
		al = get_argument(&as, cmd);
		if (al < 0) {
			r = ARCHIVE_FAILED;/* Invalid sequence. */
			goto exit_function;
		}
		if (al == 0)
			break;
		cmd += al;
		if (archive_strlen(&as) == 0 && *cmd == '\0')
			break;
		r = cmdline_add_arg(data, as.s);
		if (r != ARCHIVE_OK)
			goto exit_function;
	}
	r = ARCHIVE_OK;
exit_function:
	archive_string_free(&as);
	return (r);
}

/*
 * Set the program path.
 */
static int
cmdline_set_path(struct archive_cmdline *data, const char *path)
{
	char *newptr;

	newptr = realloc(data->path, strlen(path) + 1);
	if (newptr == NULL)
		return (ARCHIVE_FATAL);
	data->path = newptr;
	strcpy(data->path, path);
	return (ARCHIVE_OK);
}

/*
 * Add a argument for the program.
 */
static int
cmdline_add_arg(struct archive_cmdline *data, const char *arg)
{
	char **newargv;

	if (data->path == NULL)
		return (ARCHIVE_FAILED);

	newargv = realloc(data->argv, (data->argc + 2) * sizeof(char *));
	if (newargv == NULL)
		return (ARCHIVE_FATAL);
	data->argv = newargv;
	data->argv[data->argc] = strdup(arg);
	if (data->argv[data->argc] == NULL)
		return (ARCHIVE_FATAL);
	/* Set the terminator of argv. */
	data->argv[++data->argc] = NULL;
	return (ARCHIVE_OK);
}

struct archive_cmdline *
__archive_cmdline_allocate(void)
{
	return (struct archive_cmdline *)
		calloc(1, sizeof(struct archive_cmdline));
}

/*
 * Release the resources.
 */
int
__archive_cmdline_free(struct archive_cmdline *data)
{

	if (data) {
		free(data->path);
		if (data->argv != NULL) {
			int i;
			for (i = 0; data->argv[i] != NULL; i++)
				free(data->argv[i]);
			free(data->argv);
		}
		free(data);
	}
	return (ARCHIVE_OK);
}

