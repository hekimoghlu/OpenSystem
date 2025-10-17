/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
/*
 * NOTE: The contents of this file are constructed by genwrap(8); it should not
 * be manually modified.
 */

#include <sys/param.h>

#include <assert.h>
#include <err.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <regex.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <os/log.h>
#include <xpc/xpc.h>

#if defined(WRAPPER_ANALYTICS_IDENT) && !defined(WRAPPER_ANALYTICS_TESTING)
#include <CoreAnalytics/CoreAnalytics.h>
#endif
#ifdef WRAPPER_NEEDS_XCSELECT
#include <xcselect_private.h>
#endif

#ifndef nitems
#define	nitems(x)	(sizeof((x)) / sizeof((x)[0]))
#endif

/* The trailing slash is important, will be glued to WRAPPER_NAME later. */
#define	_PATH_VARSEL	"/var/select/"

#if !defined(WRAPPER_ANALYTICS_IDENT) && defined(WRAPPER_ANALYTICS_TESTING)
#error shim was improperly modified to remove the analytics identifier
#endif

/*
 * Wrappers can specify regular expressions to capture argument values.  We
 * capture those here; the expression will only be compiled exactly once, just
 * in case an argument appears multiple times.  We put the storage for that in
 * arg_expr directly, which might get kind of costly but our wrappers aren't
 * that large to begin with.
 */
struct arg_expr {
	regex_t		 expr_reg;
	const char	*expr_str;
	size_t		 expr_count;
	bool		 expr_compiled;
	bool		 expr_error;
};

/*
 * The wrapper generator will provide an array of struct application that we
 * will sift through to determine which to choose.  We'll default to the first
 * element of the array, but may choose to use one of the others depending on
 * arguments used or if the environment is set to force one.
 */
struct application {
	const char		*app_name;
	const char		*app_path;

	/*
	 * Additional arguments that the wrapper should insert when this
	 * application is selected.
	 */
	const char		 **app_add_argv;
	int			 app_add_nargv;

	/*
	 * If optstr is set, we'll use getopt(3) or getopt_long(3) to determine
	 * if we can use this application as the backend.
	 */
	const char		*app_optstr;
	const struct option	*app_longopts;

	size_t			 app_nlogonly;
	const bool		*app_logonly_opts;

	struct arg_expr		*app_shortopt_expr;
	struct arg_expr		*app_longopt_expr;

	/*
	 * Relative paths are relative to cwd, rather than to the selected
	 * developer tools.
	 */
	bool			 app_path_relcwd;

	/*
	 * Options are specified for logging purposes only, not to be considered
	 * for selecting an application to execute.
	 */
	bool			 app_opts_logonly;

	/*
	 * We may be running in a sandbox, in which case we'll denote that here
	 * to avoid too many access(2) calls.
	 */
	bool			 app_sandbox_failure;
};
