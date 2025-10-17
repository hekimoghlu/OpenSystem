/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
#if !defined(lint) && !defined(LINT)
static const char rcsid[] =
  "$FreeBSD: src/usr.sbin/cron/cron/user.c,v 1.8 1999/08/28 01:15:50 peter Exp $";
#endif

/* vix 26jan87 [log is in RCS file]
 */


#include "cron.h"

static char *User_name;

void
free_user(u)
	user	*u;
{
	entry	*e, *ne;

	free(u->name);
	for (e = u->crontab;  e != NULL;  e = ne) {
		ne = e->next;
		free_entry(e);
	}
	free(u);
}

static void
log_error(msg)
	char	*msg;
{
	log_it(User_name, getpid(), "PARSE", msg);
}

user *
load_user(crontab_fd, pw, name)
	int		crontab_fd;
	struct passwd	*pw;		/* NULL implies syscrontab */
	char		*name;
{
	char	envstr[MAX_ENVSTR];
	FILE	*file;
	user	*u;
	entry	*e;
	int	status;
	char	**envp, **tenvp;

#ifdef __APPLE__
	pw = pw;	// avoid unused parameter warning
#endif
	if (!(file = fdopen(crontab_fd, "r"))) {
		warn("fdopen on crontab_fd in load_user");
		return NULL;
	}

	Debug(DPARS, ("load_user()\n"))

	/* file is open.  build user entry, then read the crontab file.
	 */
	if ((u = (user *) malloc(sizeof(user))) == NULL) {
		errno = ENOMEM;
		return NULL;
	}
	if ((u->name = strdup(name)) == NULL) {
		free(u);
		errno = ENOMEM;
		return NULL;
	}
	u->crontab = NULL;

	/* 
	 * init environment.  this will be copied/augmented for each entry.
	 */
	if ((envp = env_init()) == NULL) {
		free(u->name);
		free(u);
		return NULL;
	}

	/*
	 * load the crontab
	 */
	while ((status = load_env(envstr, file)) >= OK) {
		switch (status) {
		case ERR:
			free_user(u);
			u = NULL;
			goto done;
		case FALSE:
			User_name = u->name;    /* for log_error */
#ifdef __APPLE__
			e = load_entry(file, log_error, strcmp(name, "*system*") ? name : NULL, envp);
#else
			e = load_entry(file, log_error, pw, envp);
#endif
			if (e) {
				e->next = u->crontab;
				u->crontab = e;
			}
			break;
		case TRUE:
			if ((tenvp = env_set(envp, envstr))) {
				envp = tenvp;
			} else {
				free_user(u);
				u = NULL;
				goto done;
			}
			break;
		}
	}

 done:
	env_free(envp);
	fclose(file);
	Debug(DPARS, ("...load_user() done\n"))
	return u;
}
