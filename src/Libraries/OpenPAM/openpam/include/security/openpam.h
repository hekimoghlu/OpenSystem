/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
#ifndef SECURITY_OPENPAM_H_INCLUDED
#define SECURITY_OPENPAM_H_INCLUDED

/*
 * Annoying but necessary header pollution
 */
#include <stdarg.h>

#include <security/openpam_attr.h>

#include <Availability.h>

#ifdef __cplusplus
extern "C" {
#endif

struct passwd;

/*
 * API extensions
 */
int
openpam_borrow_cred(pam_handle_t *_pamh,
	const struct passwd *_pwd)
	OPENPAM_NONNULL((1,2));

void
openpam_free_data(pam_handle_t *_pamh,
	void *_data,
	int _status);

void
openpam_free_envlist(char **_envlist);

const char *
openpam_get_option(pam_handle_t *_pamh,
	const char *_option);

int
openpam_restore_cred(pam_handle_t *_pamh)
	OPENPAM_NONNULL((1));

int
openpam_set_option(pam_handle_t *_pamh,
	const char *_option,
	const char *_value);

int
pam_error(const pam_handle_t *_pamh,
	const char *_fmt,
	...)
	OPENPAM_FORMAT ((__printf__, 2, 3))
	OPENPAM_NONNULL((1,2));

int
pam_get_authtok(pam_handle_t *_pamh,
	int _item,
	const char **_authtok,
	const char *_prompt)
	OPENPAM_NONNULL((1,3));

int
pam_info(const pam_handle_t *_pamh,
	const char *_fmt,
	...)
	OPENPAM_FORMAT ((__printf__, 2, 3))
	OPENPAM_NONNULL((1,2));

int
pam_prompt(const pam_handle_t *_pamh,
	int _style,
	char **_resp,
	const char *_fmt,
	...)
	OPENPAM_FORMAT ((__printf__, 4, 5))
	OPENPAM_NONNULL((1,4));

int
pam_setenv(pam_handle_t *_pamh,
	const char *_name,
	const char *_value,
	int _overwrite)
	OPENPAM_NONNULL((1,2,3));

int
pam_unsetenv(pam_handle_t *_pamh,
	const char *_name)
	OPENPAM_NONNULL((1,2));

int
pam_vinfo(const pam_handle_t *_pamh,
	const char *_fmt,
	va_list _ap)
	OPENPAM_FORMAT ((__printf__, 2, 0))
	OPENPAM_NONNULL((1,2));

int
pam_verror(const pam_handle_t *_pamh,
	const char *_fmt,
	va_list _ap)
	OPENPAM_FORMAT ((__printf__, 2, 0))
	OPENPAM_NONNULL((1,2));

int
pam_vprompt(const pam_handle_t *_pamh,
	int _style,
	char **_resp,
	const char *_fmt,
	va_list _ap)
	OPENPAM_FORMAT ((__printf__, 4, 0))
	OPENPAM_NONNULL((1,4));

/*
 * Read cooked lines.
 * Checking for _IOFBF is a fairly reliable way to detect the presence
 * of <stdio.h>, as SUSv3 requires it to be defined there.
 */
#ifdef _IOFBF
char *
openpam_readline(FILE *_f,
	int *_lineno,
	size_t *_lenp)
	OPENPAM_NONNULL((1));
#endif

/*
 * Log levels
 */
enum {
	PAM_LOG_DEBUG,
	PAM_LOG_VERBOSE,
	PAM_LOG_NOTICE,
	PAM_LOG_ERROR
};

/*
 * Log to syslog
 */
void
_openpam_log(int _level,
	const char *_func,
	const char *_fmt,
	...)
	OPENPAM_FORMAT ((__printf__, 3, 4))
	OPENPAM_NONNULL((3));

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
#define openpam_log(lvl, ...) \
	_openpam_log((lvl), __func__, __VA_ARGS__)
#elif defined(__GNUC__) && (__GNUC__ >= 3)
#define openpam_log(lvl, ...) \
	_openpam_log((lvl), __func__, __VA_ARGS__)
#elif defined(__GNUC__) && (__GNUC__ >= 2) && (__GNUC_MINOR__ >= 95)
#define openpam_log(lvl, fmt...) \
	_openpam_log((lvl), __func__, ##fmt)
#elif defined(__GNUC__) && defined(__FUNCTION__)
#define openpam_log(lvl, fmt...) \
	_openpam_log((lvl), __FUNCTION__, ##fmt)
#else
void
openpam_log(int _level,
	const char *_format,
 	...)
 	OPENPAM_FORMAT ((__printf__, 2, 3))
	OPENPAM_NONNULL((2));
#endif

/*
 * Generic conversation function
 */
struct pam_message;
struct pam_response;
int openpam_ttyconv(int _n,
	const struct pam_message **_msg,
	struct pam_response **_resp,
	void *_data);

extern int openpam_ttyconv_timeout;

/*
 * Null conversation function
 */
int openpam_nullconv(int _n,
	const struct pam_message **_msg,
	struct pam_response **_resp,
	void *_data);

/*
 * Misc conversation function
 * This function is deprecated.  Please use openpam_ttyconv instead.
 */
int misc_conv(int num_msg,
	const struct pam_message **msgm,
	struct pam_response **response,
	void *appdata_ptr) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_2,__MAC_10_6,__IPHONE_NA,__IPHONE_NA);


/*
 * PAM primitives
 */
enum {
	PAM_SM_AUTHENTICATE,
	PAM_SM_SETCRED,
	PAM_SM_ACCT_MGMT,
	PAM_SM_OPEN_SESSION,
	PAM_SM_CLOSE_SESSION,
	PAM_SM_CHAUTHTOK,
	/* keep this last */
	PAM_NUM_PRIMITIVES
};

/*
 * Dummy service module function
 */
#define PAM_SM_DUMMY(type)						\
PAM_EXTERN int								\
pam_sm_##type(pam_handle_t *pamh, int flags,				\
    int argc, const char *argv[])					\
{									\
									\
	(void)pamh;							\
	(void)flags;							\
	(void)argc;							\
	(void)argv;							\
	return (PAM_IGNORE);						\
}

/*
 * PAM service module functions match this typedef
 */
struct pam_handle;
typedef int (*pam_func_t)(struct pam_handle *, int, int, const char **);

/*
 * A struct that describes a module.
 */
typedef struct pam_module pam_module_t;
struct pam_module {
	char		*path;
	pam_func_t	 func[PAM_NUM_PRIMITIVES];
	void		*dlh;
};

/*
 * Source-code compatibility with Linux-PAM modules
 */
#if defined(PAM_SM_AUTH) || defined(PAM_SM_ACCOUNT) || \
	defined(PAM_SM_SESSION) || defined(PAM_SM_PASSWORD)
# define LINUX_PAM_MODULE
#endif

#if defined(LINUX_PAM_MODULE) && !defined(PAM_SM_AUTH)
# define _PAM_SM_AUTHENTICATE	0
# define _PAM_SM_SETCRED	0
#else
# undef PAM_SM_AUTH
# define PAM_SM_AUTH
# define _PAM_SM_AUTHENTICATE	pam_sm_authenticate
# define _PAM_SM_SETCRED	pam_sm_setcred
#endif

#if defined(LINUX_PAM_MODULE) && !defined(PAM_SM_ACCOUNT)
# define _PAM_SM_ACCT_MGMT	0
#else
# undef PAM_SM_ACCOUNT
# define PAM_SM_ACCOUNT
# define _PAM_SM_ACCT_MGMT	pam_sm_acct_mgmt
#endif

#if defined(LINUX_PAM_MODULE) && !defined(PAM_SM_SESSION)
# define _PAM_SM_OPEN_SESSION	0
# define _PAM_SM_CLOSE_SESSION	0
#else
# undef PAM_SM_SESSION
# define PAM_SM_SESSION
# define _PAM_SM_OPEN_SESSION	pam_sm_open_session
# define _PAM_SM_CLOSE_SESSION	pam_sm_close_session
#endif

#if defined(LINUX_PAM_MODULE) && !defined(PAM_SM_PASSWORD)
# define _PAM_SM_CHAUTHTOK	0
#else
# undef PAM_SM_PASSWORD
# define PAM_SM_PASSWORD
# define _PAM_SM_CHAUTHTOK	pam_sm_chauthtok
#endif

/*
 * Infrastructure for static modules using GCC linker sets.
 * You are not expected to understand this.
 */
#if defined(__FreeBSD__)
# define PAM_SOEXT ".so"
#else
# undef NO_STATIC_MODULES
# define NO_STATIC_MODULES
#endif

#if defined(__GNUC__) && !defined(__PIC__) && !defined(NO_STATIC_MODULES)
/* gcc, static linking */
# include <sys/cdefs.h>
# include <linker_set.h>
# define OPENPAM_STATIC_MODULES
# define PAM_EXTERN static
# define PAM_MODULE_ENTRY(name)						\
	static char _pam_name[] = name PAM_SOEXT;			\
	static struct pam_module _pam_module = {			\
		.path = _pam_name,					\
		.func = {						\
			[PAM_SM_AUTHENTICATE] = _PAM_SM_AUTHENTICATE,	\
			[PAM_SM_SETCRED] = _PAM_SM_SETCRED,		\
			[PAM_SM_ACCT_MGMT] = _PAM_SM_ACCT_MGMT,		\
			[PAM_SM_OPEN_SESSION] = _PAM_SM_OPEN_SESSION,	\
			[PAM_SM_CLOSE_SESSION] = _PAM_SM_CLOSE_SESSION, \
			[PAM_SM_CHAUTHTOK] = _PAM_SM_CHAUTHTOK		\
		},							\
	};								\
	DATA_SET(_openpam_static_modules, _pam_module)
#else
/* normal case */
# define PAM_EXTERN
# define PAM_MODULE_ENTRY(name)
#endif

#ifdef __cplusplus
}
#endif

#endif /* !SECURITY_OPENPAM_H_INCLUDED */
