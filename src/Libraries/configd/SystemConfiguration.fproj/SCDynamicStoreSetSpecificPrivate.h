/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 11, 2024.
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
#ifndef _SCDYNAMICSTORESETSPECIFICPRIVATE_H
#define _SCDYNAMICSTORESETSPECIFICPRIVATE_H

#include <os/availability.h>
#include <sys/cdefs.h>
#include <SystemConfiguration/SCDynamicStore.h>


/*!
	@header SCDynamicStoreSetSpecificPrivate
 */

__BEGIN_DECLS

/*!
	@function SCDynamicStoreSetConsoleInformation
	@discussion Returns information about all console users on the system.
	@param store An SCDynamicStoreRef that should be used for communication
		with the server.
		If NULL, a temporary session will be used.
	@param user A pointer to a character buffer containing the name of
		the current/primary "Console" session. If NULL, any current
		"Console" session information will be reset.
	@param uid The user ID of the current/primary "Console" user.
	@param gid The group ID of the current/primary "Console" user.
	@param sessions An array of dictionaries containing information about
		each console session on the system; NULL if no sessions are
		defined.

		The contents of this array should match that of the CoreGraphics
		CGSCopySessionList() SPI.

	@result A boolean indicating the success (or failure) of the call.
 */
Boolean
SCDynamicStoreSetConsoleInformation	(
					SCDynamicStoreRef	store,
					const char		*user,
					uid_t			uid,
					gid_t			gid,
					CFArrayRef		sessions
					)			API_AVAILABLE(macos(10.3)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);

/*!
	@function SCDynamicStoreSetConsoleUser
	@discussion Sets the name, user ID, and group ID of the currently
		logged in user.
	@param store An SCDynamicStoreRef that should be used for communication
		with the server.
		If NULL, a temporary session will be used.
	@param user A pointer to a character buffer containing the name of
		the current "Console" user. If NULL, any current "Console"
		user information will be reset.
	@param uid The user ID of the current "Console" user.
	@param gid The group ID of the current "Console" user.
	@result A boolean indicating the success (or failure) of the call.
 */
Boolean
SCDynamicStoreSetConsoleUser		(
					SCDynamicStoreRef	store,
					const char		*user,
					uid_t			uid,
					gid_t			gid
					)			API_AVAILABLE(macos(10.1)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);

__END_DECLS

#endif	/* _SCDYNAMICSTORESETSPECIFICPRIVATE_H */
