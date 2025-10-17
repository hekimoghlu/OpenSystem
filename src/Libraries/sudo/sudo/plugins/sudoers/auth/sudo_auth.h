/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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
#ifndef SUDO_AUTH_H
#define SUDO_AUTH_H

/* Auth function return values.  */
#define AUTH_SUCCESS		0
#define AUTH_FAILURE		1
#define AUTH_INTR		2
#define AUTH_FATAL		3
#define AUTH_NONINTERACTIVE	4

typedef struct sudo_auth {
    int flags;			/* various flags, see below */
    int status;			/* status from verify routine */
    const char *name;		/* name of the method as a string */
    void *data;			/* method-specific data pointer */
    int (*init)(struct passwd *pw, struct sudo_auth *auth);
    int (*setup)(struct passwd *pw, char **prompt, struct sudo_auth *auth);
    int (*verify)(struct passwd *pw, const char *p, struct sudo_auth *auth, struct sudo_conv_callback *callback);
    int (*approval)(struct passwd *pw, struct sudo_auth *auth, bool exempt);
    int (*cleanup)(struct passwd *pw, struct sudo_auth *auth, bool force);
    int (*begin_session)(struct passwd *pw, char **user_env[], struct sudo_auth *auth);
    int (*end_session)(struct passwd *pw, struct sudo_auth *auth);
} sudo_auth;

/* Values for sudo_auth.flags.  */
#define FLAG_DISABLED		0x02	/* method disabled */
#define FLAG_STANDALONE		0x04	/* standalone auth method */
#define FLAG_ONEANDONLY		0x08	/* one and only auth method */
#define FLAG_NONINTERACTIVE	0x10	/* no user input allowed */

/* Shortcuts for using the flags above. */
#define IS_DISABLED(x)		((x)->flags & FLAG_DISABLED)
#define IS_STANDALONE(x)	((x)->flags & FLAG_STANDALONE)
#define IS_ONEANDONLY(x)	((x)->flags & FLAG_ONEANDONLY)
#define IS_NONINTERACTIVE(x)	((x)->flags & FLAG_NONINTERACTIVE)

/* Like tgetpass() but uses conversation function */
char *auth_getpass(const char *prompt, int type, struct sudo_conv_callback *callback);

/* Pointer to conversation function to use with auth_getpass(). */
extern sudo_conv_t sudo_conv;

/* Prototypes for standalone methods */
int bsdauth_init(struct passwd *pw, sudo_auth *auth);
int bsdauth_verify(struct passwd *pw, const char *prompt, sudo_auth *auth, struct sudo_conv_callback *callback);
int bsdauth_approval(struct passwd *pw, sudo_auth *auth, bool exempt);
int bsdauth_cleanup(struct passwd *pw, sudo_auth *auth, bool force);
int sudo_aix_init(struct passwd *pw, sudo_auth *auth);
int sudo_aix_verify(struct passwd *pw, const char *pass, sudo_auth *auth, struct sudo_conv_callback *callback);
int sudo_aix_cleanup(struct passwd *pw, sudo_auth *auth, bool force);
int sudo_fwtk_init(struct passwd *pw, sudo_auth *auth);
int sudo_fwtk_verify(struct passwd *pw, const char *prompt, sudo_auth *auth, struct sudo_conv_callback *callback);
int sudo_fwtk_cleanup(struct passwd *pw, sudo_auth *auth, bool force);
int sudo_pam_init(struct passwd *pw, sudo_auth *auth);
int sudo_pam_init_quiet(struct passwd *pw, sudo_auth *auth);
int sudo_pam_verify(struct passwd *pw, const char *prompt, sudo_auth *auth, struct sudo_conv_callback *callback);
int sudo_pam_approval(struct passwd *pw, sudo_auth *auth, bool exempt);
int sudo_pam_cleanup(struct passwd *pw, sudo_auth *auth, bool force);
int sudo_pam_begin_session(struct passwd *pw, char **user_env[], sudo_auth *auth);
int sudo_pam_end_session(struct passwd *pw, sudo_auth *auth);
int sudo_securid_init(struct passwd *pw, sudo_auth *auth);
int sudo_securid_setup(struct passwd *pw, char **prompt, sudo_auth *auth);
int sudo_securid_verify(struct passwd *pw, const char *pass, sudo_auth *auth, struct sudo_conv_callback *callback);
int sudo_sia_setup(struct passwd *pw, char **prompt, sudo_auth *auth);
int sudo_sia_verify(struct passwd *pw, const char *prompt, sudo_auth *auth, struct sudo_conv_callback *callback);
int sudo_sia_cleanup(struct passwd *pw, sudo_auth *auth, bool force);
int sudo_sia_begin_session(struct passwd *pw, char **user_env[], sudo_auth *auth);

/* Prototypes for normal methods */
int sudo_afs_verify(struct passwd *pw, const char *pass, sudo_auth *auth, struct sudo_conv_callback *callback);
int sudo_dce_verify(struct passwd *pw, const char *pass, sudo_auth *auth, struct sudo_conv_callback *callback);
int sudo_krb5_init(struct passwd *pw, sudo_auth *auth);
int sudo_krb5_setup(struct passwd *pw, char **prompt, sudo_auth *auth);
int sudo_krb5_verify(struct passwd *pw, const char *pass, sudo_auth *auth, struct sudo_conv_callback *callback);
int sudo_krb5_cleanup(struct passwd *pw, sudo_auth *auth, bool force);
int sudo_passwd_init(struct passwd *pw, sudo_auth *auth);
int sudo_passwd_verify(struct passwd *pw, const char *pass, sudo_auth *auth, struct sudo_conv_callback *callback);
int sudo_passwd_cleanup(struct passwd *pw, sudo_auth *auth, bool force);
int sudo_rfc1938_setup(struct passwd *pw, char **prompt, sudo_auth *auth);
int sudo_rfc1938_verify(struct passwd *pw, const char *pass, sudo_auth *auth, struct sudo_conv_callback *callback);
int sudo_secureware_init(struct passwd *pw, sudo_auth *auth);
int sudo_secureware_verify(struct passwd *pw, const char *pass, sudo_auth *auth, struct sudo_conv_callback *callback);
int sudo_secureware_cleanup(struct passwd *pw, sudo_auth *auth, bool force);

/* Fields: name, flags, init, setup, verify, approval, cleanup, begin_sess, end_sess */
#define AUTH_ENTRY(n, f, i, s, v, a, c, b, e) \
	{ (f), AUTH_FAILURE, (n), NULL, (i), (s), (v), (a), (c) , (b), (e) },

#endif /* SUDO_AUTH_H */
