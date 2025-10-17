/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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
#ifndef SUDOERS_DEBUG_H
#define SUDOERS_DEBUG_H

#include "sudo_debug.h"

/*
 * Sudoers debug subsystems.
 * Note that sudoers_subsystem_ids[] is filled in at debug registration time.
 */
extern unsigned int sudoers_subsystem_ids[];
#define SUDOERS_DEBUG_ALIAS	(sudoers_subsystem_ids[ 0]) /* sudoers alias functions */
#define SUDOERS_DEBUG_AUDIT	(sudoers_subsystem_ids[ 1]) /* audit */
#define SUDOERS_DEBUG_AUTH	(sudoers_subsystem_ids[ 2]) /* authentication functions */
#define SUDOERS_DEBUG_DEFAULTS	(sudoers_subsystem_ids[ 3]) /* sudoers defaults settings */
#define SUDOERS_DEBUG_ENV	(sudoers_subsystem_ids[ 4]) /* environment handling */
#define SUDOERS_DEBUG_EVENT	(sudoers_subsystem_ids[ 5]) /* event handling */
#define SUDOERS_DEBUG_LDAP	(sudoers_subsystem_ids[ 6]) /* sudoers LDAP */
#define SUDOERS_DEBUG_LOGGING	(sudoers_subsystem_ids[ 7]) /* logging functions */
#define SUDOERS_DEBUG_MAIN	(sudoers_subsystem_ids[ 8]) /* main() */
#define SUDOERS_DEBUG_MATCH	(sudoers_subsystem_ids[ 9]) /* sudoers matching */
#define SUDOERS_DEBUG_NETIF	(sudoers_subsystem_ids[10]) /* network interface functions */
#define SUDOERS_DEBUG_NSS	(sudoers_subsystem_ids[11]) /* network service switch */
#define SUDOERS_DEBUG_PARSER	(sudoers_subsystem_ids[12]) /* sudoers parser */
#define SUDOERS_DEBUG_PERMS	(sudoers_subsystem_ids[13]) /* uid/gid swapping functions */
#define SUDOERS_DEBUG_PLUGIN	(sudoers_subsystem_ids[14]) /* main plugin functions */
#define SUDOERS_DEBUG_RBTREE	(sudoers_subsystem_ids[15]) /* red-black tree functions */
#define SUDOERS_DEBUG_SSSD	(sudoers_subsystem_ids[16]) /* sudoers SSSD */
#define SUDOERS_DEBUG_UTIL	(sudoers_subsystem_ids[17]) /* utility functions */

#endif /* SUDOERS_DEBUG_H */
