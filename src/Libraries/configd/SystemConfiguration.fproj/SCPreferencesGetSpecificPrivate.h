/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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
#ifndef _SCPREFERENCESGETSPECIFICPRIVATE_H
#define _SCPREFERENCESGETSPECIFICPRIVATE_H

#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SCPreferences.h>


/*!
	@header SCPreferencesGetSpecificPrivate
	@discussion The functions in the SCPreferencesGetSpecificPrivate API allow
		an application to get specific configuration information
		about the current system (for example, the host name).

		To access configuration preferences, you must first establish
		a preferences session using the SCPreferencesCreate function.
 */


__BEGIN_DECLS

/*!
	@function SCPreferencesGetHostName
	@discussion Gets the host name preference.
	@param prefs The preferences session.
	@result name The host name to be set;
		NULL if the name has not been set or if an error was encountered.
 */
CFStringRef
SCPreferencesGetHostName		(
					SCPreferencesRef	prefs
					);

CFStringRef
_SCPreferencesCopyLocalHostName		(
					SCPreferencesRef	prefs
					);

CFStringRef
_SCPreferencesCopyComputerName		(
					SCPreferencesRef	prefs,
					CFStringEncoding	*nameEncoding
					);
__END_DECLS

#endif /* _SCPREFERENCESGETSPECIFICPRIVATE_H */
