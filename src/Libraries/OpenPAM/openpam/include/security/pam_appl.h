/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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
#ifndef SECURITY_PAM_APPL_H_INCLUDED
#define SECURITY_PAM_APPL_H_INCLUDED

#include <security/pam_types.h>
#include <security/pam_constants.h>
#include <security/openpam_attr.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * XSSO 4.2.1, 6
 */

int
pam_acct_mgmt(pam_handle_t *_pamh,
	int _flags)
	OPENPAM_NONNULL((1));

int
pam_authenticate(pam_handle_t *_pamh,
	int _flags)
	OPENPAM_NONNULL((1));

int
pam_chauthtok(pam_handle_t *_pamh,
	int _flags)
	OPENPAM_NONNULL((1));

int
pam_close_session(pam_handle_t *_pamh,
	int _flags)
	OPENPAM_NONNULL((1));

int
pam_end(pam_handle_t *_pamh,
	int _status);

int
pam_get_data(const pam_handle_t *_pamh,
	const char *_module_data_name,
	const void **_data)
	OPENPAM_NONNULL((1,2,3));

int
pam_get_item(const pam_handle_t *_pamh,
	int _item_type,
	const void **_item)
	OPENPAM_NONNULL((1,3));

int
pam_get_user(pam_handle_t *_pamh,
	const char **_user,
	const char *_prompt)
	OPENPAM_NONNULL((1,2));

const char *
pam_getenv(pam_handle_t *_pamh,
	const char *_name)
	OPENPAM_NONNULL((1,2));

char **
pam_getenvlist(pam_handle_t *_pamh)
	OPENPAM_NONNULL((1));

int
pam_open_session(pam_handle_t *_pamh,
	int _flags)
	OPENPAM_NONNULL((1));

int
pam_putenv(pam_handle_t *_pamh,
	const char *_namevalue)
	OPENPAM_NONNULL((1,2));

int
pam_set_data(pam_handle_t *_pamh,
	const char *_module_data_name,
	void *_data,
	void (*_cleanup)(pam_handle_t *_pamh,
		void *_data,
		int _pam_end_status))
	OPENPAM_NONNULL((1,2));

int
pam_set_item(pam_handle_t *_pamh,
	int _item_type,
	const void *_item)
	OPENPAM_NONNULL((1));

int
pam_setcred(pam_handle_t *_pamh,
	int _flags)
	OPENPAM_NONNULL((1));

int
pam_start(const char *_service,
	const char *_user,
	const struct pam_conv *_pam_conv,
	pam_handle_t **_pamh)
	OPENPAM_NONNULL((4));

const char *
pam_strerror(const pam_handle_t *_pamh,
	int _error_number);

/*
 * Single Sign-On extensions
 */
#if 0
int
pam_authenticate_secondary(pam_handle_t *_pamh,
	char *_target_username,
	char *_target_module_type,
	char *_target_authn_domain,
	char *_target_supp_data,
	char *_target_module_authtok,
	int _flags);

int
pam_get_mapped_authtok(pam_handle_t *_pamh,
	const char *_target_module_username,
	const char *_target_module_type,
	const char *_target_authn_domain,
	size_t *_target_authtok_len,
	unsigned char **_target_module_authtok);

int
pam_get_mapped_username(pam_handle_t *_pamh,
	const char *_src_username,
	const char *_src_module_type,
	const char *_src_authn_domain,
	const char *_target_module_type,
	const char *_target_authn_domain,
	char **_target_module_username);

int
pam_set_mapped_authtok(pam_handle_t *_pamh,
	const char *_target_module_username,
	size_t _target_authtok_len,
	unsigned char *_target_module_authtok,
	const char *_target_module_type,
	const char *_target_authn_domain);

int
pam_set_mapped_username(pam_handle_t *_pamh,
	char *_src_username,
	char *_src_module_type,
	char *_src_authn_domain,
	char *_target_module_username,
	char *_target_module_type,
	char *_target_authn_domain);
#endif /* 0 */

#ifdef __cplusplus
}
#endif

#endif /* !SECURITY_PAM_APPL_H_INCLUDED */
