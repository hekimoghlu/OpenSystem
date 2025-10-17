/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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
#ifndef SUDOERS_LDAP_H
#define SUDOERS_LDAP_H

/* Iterators used by sudo_ldap_role_to_priv() to handle bervar ** or char ** */
typedef char * (*sudo_ldap_iter_t)(void **);

/* ldap_util.c */
bool sudo_ldap_is_negated(char **valp);
int sudo_ldap_parse_option(char *optstr, char **varp, char **valp);
struct privilege *sudo_ldap_role_to_priv(const char *cn, void *hosts, void *runasusers, void *runasgroups, void *cmnds, void *opts, const char *notbefore, const char *notafter, bool warnings, bool store_options, sudo_ldap_iter_t iter);
struct member *sudo_ldap_new_member_all(void);

#endif /* SUDOERS_LDAP_H */
