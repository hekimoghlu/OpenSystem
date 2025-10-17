/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
#ifndef SECURITY_PAM_MODULES_H_INCLUDED
#define SECURITY_PAM_MODULES_H_INCLUDED

#include <security/pam_types.h>
#include <security/pam_constants.h>
#include <security/openpam.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * XSSO 4.2.2, 6
 */

#if defined(PAM_SM_ACCOUNT)
PAM_EXTERN int
pam_sm_acct_mgmt(pam_handle_t *_pamh,
	int _flags,
	int _argc,
	const char **_argv);
#endif

#if defined(PAM_SM_AUTH)
PAM_EXTERN int
pam_sm_authenticate(pam_handle_t *_pamh,
	int _flags,
	int _argc,
	const char **_argv);
#endif

#if defined(PAM_SM_PASSWORD)
PAM_EXTERN int
pam_sm_chauthtok(pam_handle_t *_pamh,
	int _flags,
	int _argc,
	const char **_argv);
#endif

#if defined(PAM_SM_SESSION)
PAM_EXTERN int
pam_sm_close_session(pam_handle_t *_pamh,
	int _flags,
	int _args,
	const char **_argv);
#endif

#if defined(PAM_SM_SESSION)
PAM_EXTERN int
pam_sm_open_session(pam_handle_t *_pamh,
	int _flags,
	int _argc,
	const char **_argv);
#endif

#if defined(PAM_SM_AUTH)
PAM_EXTERN int
pam_sm_setcred(pam_handle_t *_pamh,
	int _flags,
	int _argc,
	const char **_argv);
#endif

/*
 * Single Sign-On extensions
 */
#if 0
PAM_EXTERN int
pam_sm_authenticate_secondary(pam_handle_t *_pamh,
	char *_target_username,
	char *_target_module_type,
	char *_target_authn_domain,
	char *_target_supp_data,
	unsigned char *_target_module_authtok,
	int _flags,
	int _argc,
	const char **_argv);

PAM_EXTERN int
pam_sm_get_mapped_authtok(pam_handle_t *_pamh,
	char *_target_module_username,
	char *_target_module_type,
	char *_target_authn_domain,
	size_t *_target_authtok_len,
	unsigned char **_target_module_authtok,
	int _argc,
	char *_argv);

PAM_EXTERN int
pam_sm_get_mapped_username(pam_handle_t *_pamh,
	char *_src_username,
	char *_src_module_type,
	char *_src_authn_domain,
	char *_target_module_type,
	char *_target_authn_domain,
	char **_target_module_username,
	int _argc,
	const char **_argv);

PAM_EXTERN int
pam_sm_set_mapped_authtok(pam_handle_t *_pamh,
	char *_target_module_username,
	size_t _target_authtok_len,
	unsigned char *_target_module_authtok,
	char *_target_module_type,
	char *_target_authn_domain,
	int _argc,
	const char *_argv);

PAM_EXTERN int
pam_sm_set_mapped_username(pam_handle_t *_pamh,
	char *_target_module_username,
	char *_target_module_type,
	char *_target_authn_domain,
	int _argc,
	const char **_argv);

#endif /* 0 */

#ifdef __cplusplus
}
#endif

#endif /* !SECURITY_PAM_MODULES_H_INCLUDED */
