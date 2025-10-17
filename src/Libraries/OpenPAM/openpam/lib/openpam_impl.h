/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
#ifndef _OPENPAM_IMPL_H_INCLUDED
#define _OPENPAM_IMPL_H_INCLUDED

#define PAM_LOG_LIBDEBUG -1

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <security/openpam.h>

extern const char *_pam_func_name[PAM_NUM_PRIMITIVES];
extern const char *_pam_sm_func_name[PAM_NUM_PRIMITIVES];
extern const char *_pam_err_name[PAM_NUM_ERRORS];
extern const char *_pam_apple_err_name[PAM_APPLE_NUM_ERRORS];
extern const char *_pam_item_name[PAM_NUM_ITEMS];

extern int _openpam_debug;

/*
 * Control flags
 */
typedef enum {
	PAM_BINDING,
	PAM_REQUIRED,
	PAM_REQUISITE,
	PAM_SUFFICIENT,
	PAM_OPTIONAL,
	PAM_NUM_CONTROL_FLAGS
} pam_control_t;

/*
 * Facilities
 */
typedef enum {
	PAM_FACILITY_ANY = -1,
	PAM_AUTH = 0,
	PAM_ACCOUNT,
	PAM_SESSION,
	PAM_PASSWORD,
	PAM_NUM_FACILITIES
} pam_facility_t;

typedef struct pam_chain pam_chain_t;
struct pam_chain {
	pam_module_t	*module;
	int		 flag;
	int		 optc;
	char	       **optv;
	pam_chain_t	*next;
};

typedef struct pam_data pam_data_t;
struct pam_data {
	char		*name;
	void		*data;
	void		(*cleanup)(pam_handle_t *, void *, int);
	pam_data_t	*next;
};

struct pam_handle {
	char		*service;

	/* chains */
	pam_chain_t	*chains[PAM_NUM_FACILITIES];
	pam_chain_t	*current;
	int		 primitive;

	/* items and data */
	void		*item[PAM_NUM_ITEMS];
	pam_data_t	*module_data;

	/* environment list */
	char	       **env;
	int		 env_count;
	int		 env_size;
};

#ifdef NGROUPS_MAX
#define PAM_SAVED_CRED "pam_saved_cred"
struct pam_saved_cred {
	uid_t	 euid;
	gid_t	 egid;
	gid_t	 groups[NGROUPS_MAX];
	int	 ngroups;
};
#endif

#define PAM_OTHER	"other"

int		 openpam_configure(pam_handle_t *, const char *);
int		 openpam_configure_default(pam_handle_t *);
int		 openpam_dispatch(pam_handle_t *, int, int);
int		 openpam_findenv(pam_handle_t *, const char *, size_t);
pam_module_t	*openpam_load_module(const char *);
void			openpam_release_module(pam_module_t *module);
void		 openpam_clear_chains(pam_chain_t **);

typedef enum { pam_conf_style, pam_d_style } openpam_style_t;

int		 openpam_configure_apple(pam_handle_t *, const char *);

#ifdef OPENPAM_STATIC_MODULES
pam_module_t	*openpam_static(const char *);
#endif
pam_module_t	*openpam_dynamic(const char *);

#define	FREE(p) do { free((p)); (p) = NULL; } while (0)

#ifdef DEBUG
#define ENTER() openpam_log(PAM_LOG_LIBDEBUG, "entering")
#define ENTERI(i) do { \
	int _i = (i); \
	if (_i > 0 && _i < PAM_NUM_ITEMS) \
		openpam_log(PAM_LOG_LIBDEBUG, "entering: %s", _pam_item_name[_i]); \
	else \
		openpam_log(PAM_LOG_LIBDEBUG, "entering: %d", _i); \
} while (0)
#define ENTERN(n) do { \
	int _n = (n); \
	openpam_log(PAM_LOG_LIBDEBUG, "entering: %d", _n); \
} while (0)
#define ENTERS(s) do { \
	const char *_s = (s); \
	if (_s == NULL) \
		openpam_log(PAM_LOG_LIBDEBUG, "entering: NULL"); \
	else \
		openpam_log(PAM_LOG_LIBDEBUG, "entering: '%s'", _s); \
} while (0)
#define	RETURNV() openpam_log(PAM_LOG_LIBDEBUG, "returning")
#define RETURNC(c) do { \
	int _c = (c); \
	if (_c >= 0 && _c < PAM_NUM_ERRORS) \
		openpam_log(PAM_LOG_LIBDEBUG, "returning %s", _pam_err_name[_c]); \
	else if (_c >= PAM_APPLE_MIN_ERROR && _c < PAM_APPLE_MAX_ERROR) \
		openpam_log(PAM_LOG_LIBDEBUG, "returning %s", _pam_apple_err_name[_c-PAM_APPLE_MIN_ERROR]); \
	else \
		openpam_log(PAM_LOG_LIBDEBUG, "returning %d!", _c); \
	return (_c); \
} while (0)
#define	RETURNN(n) do { \
	int _n = (n); \
	openpam_log(PAM_LOG_LIBDEBUG, "returning %d", _n); \
	return (_n); \
} while (0)
#define	RETURNP(p) do { \
	const void *_p = (p); \
	if (_p == NULL) \
		openpam_log(PAM_LOG_LIBDEBUG, "returning NULL"); \
	else \
		openpam_log(PAM_LOG_LIBDEBUG, "returning %p", _p); \
	return (p); \
} while (0)
#define	RETURNS(s) do { \
	const char *_s = (s); \
	if (_s == NULL) \
		openpam_log(PAM_LOG_LIBDEBUG, "returning NULL"); \
	else \
		openpam_log(PAM_LOG_LIBDEBUG, "returning '%s'", _s); \
	return (_s); \
} while (0)
#else
#define ENTER()
#define ENTERI(i)
#define ENTERN(n)
#define ENTERS(s)
#define RETURNV() return
#define RETURNC(c) return (c)
#define RETURNN(n) return (n)
#define RETURNP(p) return (p)
#define RETURNS(s) return (s)
#endif

#endif
