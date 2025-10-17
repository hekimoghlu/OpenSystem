/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
#ifndef __login_protos_h__
#define __login_protos_h__

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

void
add_env (
	const char */*var*/,
	const char */*value*/);

void
check_shadow (
	const struct passwd */*pw*/,
	const struct spwd */*sp*/);

char *
clean_ttyname (char */*tty*/);

void
copy_env (void);

int
do_osfc2_magic (uid_t /*uid*/);

void
extend_env (char */*str*/);

int
login_access (
	struct passwd */*user*/,
	char */*from*/);

char *
login_conf_get_string (const char */*str*/);

void
login_read_env (const char */*file*/);

char *
make_id (char */*tty*/);

void
prepare_utmp (
	struct utmp */*utmp*/,
	char */*tty*/,
	const char */*username*/,
	const char */*hostname*/);

int
read_limits_conf (
	const char */*file*/,
	const struct passwd */*pwd*/);

int
read_string (
	const char */*prompt*/,
	char */*buf*/,
	size_t /*len*/,
	int /*echo*/);

void
shrink_hostname (
	const char */*hostname*/,
	char */*dst*/,
	size_t /*dst_sz*/);

void
stty_default (void);

void
utmp_login (
	char */*tty*/,
	const char */*username*/,
	const char */*hostname*/);

int
utmpx_login (
	char */*line*/,
	const char */*user*/,
	const char */*host*/);

#ifdef __cplusplus
}
#endif

#endif /* __login_protos_h__ */
