/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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
#ifndef _SCPREFERENCESSETSPECIFIC_H
#define _SCPREFERENCESSETSPECIFIC_H

#include <os/availability.h>
#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SCPreferences.h>

CF_IMPLICIT_BRIDGING_ENABLED
CF_ASSUME_NONNULL_BEGIN

/*!
	@header SCPreferencesSetSpecific
	@discussion The functions in the SCPreferencesSetSpecific API allow
		an application to set specific configuration information
		about the current system (for example, the computer or
		sharing name).

		To access configuration preferences, you must first establish
		a preferences session using the SCPreferencesCreate function.
 */


__BEGIN_DECLS

/*!
	@function SCPreferencesSetComputerName
	@discussion Updates the computer name preference.

		Note: To commit these changes to permanent storage you must
		call the SCPreferencesCommitChanges function.
		In addition, you must call the SCPreferencesApplyChanges
		function for the new name to become active.
	@param prefs The preferences session.
	@param name The computer name to be set.
	@param nameEncoding The encoding associated with the computer name.
	@result Returns TRUE if successful; FALSE otherwise.
 */
Boolean
SCPreferencesSetComputerName		(
					SCPreferencesRef	prefs,
					CFStringRef __nullable	name,
					CFStringEncoding	nameEncoding
					)			API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

/*!
	@function SCPreferencesSetLocalHostName
	@discussion Updates the local host name.

		Note: To commit these changes to permanent storage you must
		call the SCPreferencesCommitChanges function.
		In addition, you must call the SCPreferencesApplyChanges
		function for the new name to become active.
	@param prefs The preferences session.
	@param name The local host name to be set.

	Note: this string must conform to the naming conventions of a DNS host
		name as specified in RFC 1034 (section 3.5).
	@result Returns TRUE if successful; FALSE otherwise.
 */
Boolean
SCPreferencesSetLocalHostName		(
					SCPreferencesRef	prefs,
					CFStringRef __nullable	name
					)			API_AVAILABLE(macos(10.2)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

__END_DECLS

CF_ASSUME_NONNULL_END
CF_IMPLICIT_BRIDGING_DISABLED

#endif	/* _SCPREFERENCESSETSPECIFIC_H */
