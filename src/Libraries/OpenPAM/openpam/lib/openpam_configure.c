/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <System/sys/codesign.h>	/* csops() */
#include <security/pam_appl.h>

#include "openpam_impl.h"

const char *_pam_facility_name[PAM_NUM_FACILITIES] = {
	[PAM_ACCOUNT]		= "account",
	[PAM_AUTH]		= "auth",
	[PAM_PASSWORD]		= "password",
	[PAM_SESSION]		= "session",
};

const char *_pam_control_flag_name[PAM_NUM_CONTROL_FLAGS] = {
	[PAM_BINDING]		= "binding",
	[PAM_OPTIONAL]		= "optional",
	[PAM_REQUIRED]		= "required",
	[PAM_REQUISITE]		= "requisite",
	[PAM_SUFFICIENT]	= "sufficient",
};

static int openpam_load_chain(pam_handle_t *, const char *, pam_facility_t);

/*
 * Matches a word against the first one in a string.
 * Returns non-zero if they match.
 */
static int
match_word(const char *str, const char *word)
{
	while (*str && tolower(*str) == tolower(*word))
		++str, ++word;
	return (*str == ' ' && *word == '\0');
}

/*
 * Return a pointer to the next word (or the final NUL) in a string.
 */
static const char *
next_word(const char *str)
{
	/* skip current word */
	while (*str && *str != ' ')
		++str;
	/* skip whitespace */
	while (*str == ' ')
		++str;
	return (str);
}

/*
 * Return a malloc()ed copy of the first word in a string.
 */
static char *
dup_word(const char *str)
{
	const char *end;
	char *word;

	for (end = str; *end && *end != ' '; ++end)
		/* nothing */ ;
	if (asprintf(&word, "%.*s", (int)(end - str), str) < 0)
		return (NULL);
	return (word);
}

/*
 * Return the length of the first word in a string.
 */
static int
wordlen(const char *str)
{
	int i;

	for (i = 0; str[i] && str[i] != ' '; ++i)
		/* nothing */ ;
	return (i);
}

/*
 * Extracts given chains from a policy file.
 */

int
openpam_read_chain_from_filehandle(pam_handle_t *pamh,
	const char *service,
	pam_facility_t facility,
	FILE *f,				/* consumes f as in: fclose(f); */
	const char *filename,
	openpam_style_t style)
{
	pam_chain_t *this, **next;
	const char *p, *q;
	int count, i, lineno, ret;
	pam_facility_t fclt;
	pam_control_t ctlf;
	char *line, *name;

	this = NULL;
	count = lineno = 0;
	while ((line = openpam_readline(f, &lineno, NULL)) != NULL) {
		p = line;

		/* match service name */
		if (style == pam_conf_style) {
			if (!match_word(p, service)) {
				FREE(line);
				continue;
			}
			p = next_word(p);
		}

		/* match facility name */
		for (fclt = 0; fclt < PAM_NUM_FACILITIES; ++fclt)
			if (match_word(p, _pam_facility_name[fclt]))
				break;
		if (fclt == PAM_NUM_FACILITIES) {
			openpam_log(PAM_LOG_NOTICE,
			    "%s(%d): invalid facility '%.*s' (ignored)",
			    filename, lineno, wordlen(p), p);
			goto fail;
		}
		if (facility != fclt && facility != PAM_FACILITY_ANY) {
			FREE(line);
			continue;
		}
		p = next_word(p);

		/* include other chain */
		if (match_word(p, "include")) {
			p = next_word(p);
			if (*next_word(p) != '\0')
				openpam_log(PAM_LOG_NOTICE,
				    "%s(%d): garbage at end of 'include' line",
				    filename, lineno);
			if ((name = dup_word(p)) == NULL)
				goto syserr;
			ret = openpam_load_chain(pamh, name, fclt);
			FREE(name);
			if (ret < 0)
				goto fail;
			count += ret;
			FREE(line);
			continue;
		}

		/* allocate new entry */
		if ((this = calloc(1, sizeof *this)) == NULL)
			goto syserr;

		/* control flag */
		for (ctlf = 0; ctlf < PAM_NUM_CONTROL_FLAGS; ++ctlf)
			if (match_word(p, _pam_control_flag_name[ctlf]))
				break;
		if (ctlf == PAM_NUM_CONTROL_FLAGS) {
			openpam_log(PAM_LOG_ERROR,
			    "%s(%d): invalid control flag '%.*s'",
			    filename, lineno, wordlen(p), p);
			goto fail;
		}
		this->flag = ctlf;

		/* module name */
		p = next_word(p);
		if (*p == '\0') {
			openpam_log(PAM_LOG_ERROR,
			    "%s(%d): missing module name",
			    filename, lineno);
			goto fail;
		}
		if ((name = dup_word(p)) == NULL)
			goto syserr;
		this->module = openpam_load_module(name);
		FREE(name);
		if (this->module == NULL)
			goto fail;

		/* module options */
		p = q = next_word(p);
		while (*q != '\0') {
			++this->optc;
			q = next_word(q);
		}
		this->optv = calloc(this->optc + 1, sizeof(char *));
		if (this->optv == NULL)
			goto syserr;
		for (i = 0; i < this->optc; ++i) {
			if ((this->optv[i] = dup_word(p)) == NULL)
				goto syserr;
			p = next_word(p);
		}

		/* hook it up */
		for (next = &pamh->chains[fclt]; *next != NULL;
		     next = &(*next)->next)
			/* nothing */ ;
		*next = this;
		this = NULL;
		++count;

		/* next please... */
		FREE(line);
	}
	if (!feof(f))
		goto syserr;
	fclose(f);
	return (count);
 syserr:
	openpam_log(PAM_LOG_ERROR, "%s: %m", filename);
 fail:
	FREE(this);
	FREE(line);
	fclose(f);
	return (-1);
}

static int
openpam_read_chain_from_path(pam_handle_t *pamh,
	const char *service,
	pam_facility_t facility,
	const char *filename,
	openpam_style_t style)
{
	FILE *f = fopen(filename, "r");
	if (f == NULL) {
		openpam_log(errno == ENOENT ? PAM_LOG_LIBDEBUG : PAM_LOG_NOTICE,
					"%s: %m", filename);
		return (errno == EPERM ? -1 : 0);
	}

	return openpam_read_chain_from_filehandle(pamh, service, facility, f, filename, style);
}

static const char *openpam_policy_path[] = {
#ifdef __APPLE_MDM_SUPPORT__
    "/private/var/db/ManagedConfigurationFiles/com.apple.pam/etc/pam.d/",
    "/private/var/db/ManagedConfigurationFiles/com.apple.pam/etc/pam.conf",
#endif // __APPLE_MDM_SUPPORT__
	"/etc/pam.d/",
	"/etc/pam.conf",
	"/usr/local/etc/pam.d/",
	"/usr/local/etc/pam.conf",
	NULL
};

/*
 * Locates the policy file for a given service and reads the given chains
 * from it.
 */
static int
openpam_load_chain(pam_handle_t *pamh,
	const char *service,
	pam_facility_t facility)
{
	const char **path;
	char *filename;
	size_t len;
	int r;

	/* don't allow to escape from policy_path */
	if (strchr(service, '/')) {
		openpam_log(PAM_LOG_ERROR, "invalid service name: %s",
		    service);
		return (-PAM_SYSTEM_ERR);
	}

	for (path = openpam_policy_path; *path != NULL; ++path) {
		len = strlen(*path);
		if ((*path)[len - 1] == '/') {
			if (asprintf(&filename, "%s%s", *path, service) < 0) {
				openpam_log(PAM_LOG_ERROR, "asprintf(): %m");
				return (-PAM_BUF_ERR);
			}
			r = openpam_read_chain_from_path(pamh, service, facility,
			    filename, pam_d_style);
			FREE(filename);
		} else {
			r = openpam_read_chain_from_path(pamh, service, facility,
			    *path, pam_conf_style);
		}
		if (r != 0)
			return (r);
	}
	return (0);
}

/*
 * <rdar://problem/27991863> Sandbox apps report all passwords as valid
 * Default all empty facilities to "required pam_deny.so"
 */
int
openpam_configure_default(pam_handle_t *pamh)
{
	pam_facility_t fclt;

	for (fclt = 0; fclt < PAM_NUM_FACILITIES; ++fclt) {
		if (pamh->chains[fclt] == NULL) {
			pam_chain_t *this = calloc(1, sizeof(pam_chain_t));
			if (this == NULL)
				goto load_err;
			this->flag   = PAM_REQUIRED;
			this->module = openpam_load_module("pam_deny.so");
		/*	this->optc   = 0;	*/
			this->optv   = calloc(1, sizeof(char *));
		/*	this->next	 = NULL;	*/
			if (this->optv != NULL && this->module != NULL) {
				pamh->chains[fclt] = this;
			} else {
				if (this->optv != NULL)
					free(this->optv);
				if (this->module != NULL)
					openpam_release_module(this->module);
				free(this);
				goto load_err;
			}
		}
	}
	return (PAM_SUCCESS);

load_err:
	openpam_clear_chains(pamh->chains);
	return (PAM_SYSTEM_ERR);
}

/*
 * OpenPAM internal
 *
 * Configure a service
 */

int
openpam_configure(pam_handle_t *pamh,
	const char *service)
{
	pam_facility_t fclt;

	if (openpam_load_chain(pamh, service, PAM_FACILITY_ANY) < 0)
		goto load_err;

	for (fclt = 0; fclt < PAM_NUM_FACILITIES; ++fclt) {
		if (pamh->chains[fclt] != NULL)
			continue;
		if (openpam_load_chain(pamh, PAM_OTHER, fclt) < 0)
			goto load_err;
	}
	if (openpam_configure_default(pamh))
		goto load_err;
	return (PAM_SUCCESS);

 load_err:
	openpam_clear_chains(pamh->chains);
	return (PAM_SYSTEM_ERR);
}

/*
 * NODOC
 *
 * Error codes:
 *	PAM_SYSTEM_ERR
 */
