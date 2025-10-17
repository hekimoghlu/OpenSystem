/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
#include <sys/types.h>

#include <err.h>
#include <errno.h>
#ifndef __APPLE__
#include <login_cap.h>
#endif
#include <pwd.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "envopts.h"

extern char **environ;

int	 env_verbosity;

static void usage(void) __dead2;

/*
 * Exit codes.
 */
#define EXIT_CANCELED      125 /* Internal error prior to exec attempt. */
#define EXIT_CANNOT_INVOKE 126 /* Program located, but not usable. */
#define EXIT_ENOENT        127 /* Could not find program to exec. */

int
main(int argc, char **argv)
{
	char *altpath, *altwd, **ep, *p, **parg, term;
	char *cleanenv[1];
#ifndef __APPLE__
	char *login_class, *login_name;
	struct passwd *pw;
	login_cap_t *lc;
	bool login_as_user;
	uid_t uid;
#endif
	int ch, want_clear;
	int rtrn;

	altpath = NULL;
	altwd = NULL;
#ifndef __APPLE__
	login_class = NULL;
	login_name = NULL;
	pw = NULL;
	lc = NULL;
	login_as_user = false;
#endif
	want_clear = 0;
	term = '\n';
#ifdef __APPLE__
	while ((ch = getopt(argc, argv, "-0C:iP:S:u:v")) != -1)
#else
	while ((ch = getopt(argc, argv, "-0C:iL:P:S:U:u:v")) != -1)
#endif
		switch(ch) {
		case '-':
		case 'i':
			want_clear = 1;
			break;
		case '0':
			term = '\0';
			break;
		case 'C':
			altwd = optarg;
			break;
#ifndef __APPLE__
		case 'U':
			login_as_user = true;
			/* FALLTHROUGH */
		case 'L':
			login_name = optarg;
			break;
#endif
		case 'P':
			altpath = optarg;
			break;
		case 'S':
			/*
			 * The -S option, for "split string on spaces, with
			 * support for some simple substitutions"...
			 */
			split_spaces(optarg, &optind, &argc, &argv);
			break;
		case 'u':
			if (env_verbosity)
				fprintf(stderr, "#env unset:\t%s\n", optarg);
			rtrn = unsetenv(optarg);
			if (rtrn == -1)
				err(EXIT_FAILURE, "unsetenv %s", optarg);
			break;
		case 'v':
			env_verbosity++;
			if (env_verbosity > 1)
				fprintf(stderr, "#env verbosity now at %d\n",
				    env_verbosity);
			break;
		case '?':
		default:
			usage();
		}
	if (want_clear) {
		environ = cleanenv;
		cleanenv[0] = NULL;
		if (env_verbosity)
			fprintf(stderr, "#env clearing environ\n");
	}
#ifndef __APPLE__
	if (login_name != NULL) {
		login_class = strchr(login_name, '/');
		if (login_class)
			*login_class++ = '\0';
		if (*login_name != '\0' && strcmp(login_name, "-") != 0) {
			pw = getpwnam(login_name);
			if (pw == NULL) {
				char *endp = NULL;
				errno = 0;
				uid = strtoul(login_name, &endp, 10);
				if (errno == 0 && *endp == '\0')
					pw = getpwuid(uid);
			}
			if (pw == NULL)
				errx(EXIT_FAILURE, "no such user: %s", login_name);
		}
		/*
		 * Note that it is safe for pw to be null here; the libutil
		 * code handles that, bypassing substitution of $ and using
		 * the class "default" if no class name is given either.
		 */
		if (login_class != NULL) {
			lc = login_getclass(login_class);
			if (lc == NULL)
				errx(EXIT_FAILURE, "no such login class: %s",
				    login_class);
		} else {
			lc = login_getpwclass(pw);
			if (lc == NULL)
				errx(EXIT_FAILURE, "login_getpwclass failed");
		}

		/*
		 * This is not done with setusercontext() because that will
		 * try and use ~/.login_conf even when we don't want it to.
		 */
		setclassenvironment(lc, pw, 1);
		setclassenvironment(lc, pw, 0);
		if (login_as_user) {
			login_close(lc);
			if ((lc = login_getuserclass(pw)) != NULL) {
				setclassenvironment(lc, pw, 1);
				setclassenvironment(lc, pw, 0);
			}
		}
		endpwent();
		if (lc != NULL)
			login_close(lc);
	}
#endif
	for (argv += optind; *argv && (p = strchr(*argv, '=')); ++argv) {
		if (env_verbosity)
			fprintf(stderr, "#env setenv:\t%s\n", *argv);
		*p = '\0';
		rtrn = setenv(*argv, p + 1, 1);
		*p = '=';
		if (rtrn == -1)
			err(EXIT_FAILURE, "setenv %s", *argv);
	}
	if (*argv) {
		if (term == '\0')
			errx(EXIT_CANCELED, "cannot specify command with -0");
		if (altwd && chdir(altwd) != 0)
			err(EXIT_CANCELED, "cannot change directory to '%s'",
			    altwd);
		if (altpath)
			search_paths(altpath, argv);
		if (env_verbosity) {
			fprintf(stderr, "#env executing:\t%s\n", *argv);
			for (parg = argv, argc = 0; *parg; parg++, argc++)
				fprintf(stderr, "#env    arg[%d]=\t'%s'\n",
				    argc, *parg);
			if (env_verbosity > 1)
				sleep(1);
		}
		execvp(*argv, argv);
		err(errno == ENOENT ? EXIT_ENOENT : EXIT_CANNOT_INVOKE,
		    "%s", *argv);
	} else {
		if (altwd)
			errx(EXIT_CANCELED, "must specify command with -C");
		if (altpath)
			errx(EXIT_CANCELED, "must specify command with -P");
	}
	for (ep = environ; *ep; ep++)
		(void)printf("%s%c", *ep, term);
	if (fflush(stdout) != 0)
		err(1, "stdout");
	exit(0);
}

static void
usage(void)
{
	(void)fprintf(stderr,
#ifdef __APPLE__
	    "usage: env [-0iv] [-C workdir] [-P utilpath] [-S string]\n"
#else
	    "usage: env [-0iv] [-C workdir] [-L|-U user[/class]] [-P utilpath] [-S string]\n"
#endif
	    "           [-u name] [name=value ...] [utility [argument ...]]\n");
	exit(1);
}
