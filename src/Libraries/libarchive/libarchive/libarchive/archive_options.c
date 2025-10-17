/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#include "archive_options_private.h"

static const char *
parse_option(const char **str,
    const char **mod, const char **opt, const char **val);

int
_archive_set_option(struct archive *a,
    const char *m, const char *o, const char *v,
    int magic, const char *fn, option_handler use_option)
{
	const char *mp, *op, *vp;
	int r;

	archive_check_magic(a, magic, ARCHIVE_STATE_NEW, fn);

	mp = (m != NULL && m[0] != '\0') ? m : NULL;
	op = (o != NULL && o[0] != '\0') ? o : NULL;
	vp = (v != NULL && v[0] != '\0') ? v : NULL;

	if (op == NULL && vp == NULL)
		return (ARCHIVE_OK);
	if (op == NULL) {
		archive_set_error(a, ARCHIVE_ERRNO_MISC, "Empty option");
		return (ARCHIVE_FAILED);
	}

	r = use_option(a, mp, op, vp);
	if (r == ARCHIVE_WARN - 1) {
		archive_set_error(a, ARCHIVE_ERRNO_MISC,
		    "Unknown module name: `%s'", mp);
		return (ARCHIVE_FAILED);
	}
	if (r == ARCHIVE_WARN) {
		archive_set_error(a, ARCHIVE_ERRNO_MISC,
		    "Undefined option: `%s%s%s%s%s%s'",
		    vp?"":"!", mp?mp:"", mp?":":"", op, vp?"=":"", vp?vp:"");
		return (ARCHIVE_FAILED);
	}
	return (r);
}

int
_archive_set_either_option(struct archive *a, const char *m, const char *o, const char *v,
    option_handler use_format_option, option_handler use_filter_option)
{
	int r1, r2;

	if (o == NULL && v == NULL)
		return (ARCHIVE_OK);
	if (o == NULL)
		return (ARCHIVE_FAILED);

	r1 = use_format_option(a, m, o, v);
	if (r1 == ARCHIVE_FATAL)
		return (ARCHIVE_FATAL);

	r2 = use_filter_option(a, m, o, v);
	if (r2 == ARCHIVE_FATAL)
		return (ARCHIVE_FATAL);

	if (r2 == ARCHIVE_WARN - 1)
		return r1;
	return r1 > r2 ? r1 : r2;
}

int
_archive_set_options(struct archive *a, const char *options,
    int magic, const char *fn, option_handler use_option)
{
	int allok = 1, anyok = 0, ignore_mod_err = 0, r;
	char *data;
	const char *s, *mod, *opt, *val;

	archive_check_magic(a, magic, ARCHIVE_STATE_NEW, fn);

	if (options == NULL || options[0] == '\0')
		return ARCHIVE_OK;

	if ((data = strdup(options)) == NULL) {
		archive_set_error(a,
		    ENOMEM, "Out of memory adding file to list");
		return (ARCHIVE_FATAL);
	}
	s = (const char *)data;

	do {
		mod = opt = val = NULL;

		parse_option(&s, &mod, &opt, &val);
		if (mod == NULL && opt != NULL &&
		    strcmp("__ignore_wrong_module_name__", opt) == 0) {
			/* Ignore module name error */
			if (val != NULL) {
				ignore_mod_err = 1;
				anyok = 1;
			}
			continue;
		}

		r = use_option(a, mod, opt, val);
		if (r == ARCHIVE_FATAL) {
			free(data);
			return (ARCHIVE_FATAL);
		}
		if (r == ARCHIVE_FAILED && mod != NULL) {
			free(data);
			return (ARCHIVE_FAILED);
		}
		if (r == ARCHIVE_WARN - 1) {
			if (ignore_mod_err)
				continue;
			/* The module name is wrong. */
			archive_set_error(a, ARCHIVE_ERRNO_MISC,
			    "Unknown module name: `%s'", mod);
			free(data);
			return (ARCHIVE_FAILED);
		}
		if (r == ARCHIVE_WARN) {
			/* The option name is wrong. No-one used this. */
			archive_set_error(a, ARCHIVE_ERRNO_MISC,
			    "Undefined option: `%s%s%s'",
			    mod?mod:"", mod?":":"", opt);
			free(data);
			return (ARCHIVE_FAILED);
		}
		if (r == ARCHIVE_OK)
			anyok = 1;
		else
			allok = 0;
	} while (s != NULL);

	free(data);
	return allok ? ARCHIVE_OK : anyok ? ARCHIVE_WARN : ARCHIVE_FAILED;
}

static const char *
parse_option(const char **s, const char **m, const char **o, const char **v)
{
	const char *end, *mod, *opt, *val;
	char *p;

	end = NULL;
	mod = NULL;
	opt = *s;
	val = "1";

	p = strchr(opt, ',');

	if (p != NULL) {
		*p = '\0';
		end = ((const char *)p) + 1;
	}

	if (0 == strlen(opt)) {
		*s = end;
		*m = NULL;
		*o = NULL;
		*v = NULL;
		return end;
	}

	p = strchr(opt, ':');
	if (p != NULL) {
		*p = '\0';
		mod = opt;
		opt = ++p;
	}

	p = strchr(opt, '=');
	if (p != NULL) {
		*p = '\0';
		val = ++p;
	} else if (opt[0] == '!') {
		++opt;
		val = NULL;
	}

	*s = end;
	*m = mod;
	*o = opt;
	*v = val;

	return end;
}

