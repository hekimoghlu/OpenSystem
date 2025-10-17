/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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
#ifndef _SCPREFERENCESSETSPECIFICPRIVATE_H
#define _SCPREFERENCESSETSPECIFICPRIVATE_H

#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SCPreferences.h>


/*!
	@header SCPreferencesSetSpecificPrivate
	@discussion The functions in the SCPreferencesSetSpecificPrivate API allow
		an application to set specific configuration information
		about the current system (for example, the computer or
		sharing name).

		To access configuration preferences, you must first establish
		a preferences session using the SCPreferencesCreate function.
 */


__BEGIN_DECLS

/*!
	@function SCPreferencesSetHostName
	@discussion Updates the host name preference.

		Note: To commit these changes to permanent storage you must
		call the SCPreferencesCommitChanges function.
		In addition, you must call the SCPreferencesApplyChanges
		function for the new name to become active.
	@param prefs The preferences session.
	@param name The host name to be set.
	@result Returns TRUE if successful; FALSE otherwise.
 */
Boolean
SCPreferencesSetHostName		(
					SCPreferencesRef	prefs,
					CFStringRef		name
					);

__END_DECLS

#endif /* _SCPREFERENCESSETSPECIFICPRIVATE_H */
