/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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
#ifndef _SCDYNAMICSTORECOPYSPECIFICPRIVATE_H
#define _SCDYNAMICSTORECOPYSPECIFICPRIVATE_H

#include <os/availability.h>
#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SCDynamicStore.h>


/*!
	@header SCDynamicStoreCopySpecificPrivate
	@discussion The following APIs allow an application to retrieve
		console information.
 */


__BEGIN_DECLS

/*
 * Predefined keys for the console session dictionaries
 */
extern const CFStringRef kSCConsoleSessionID			/* value is CFNumber */
		API_AVAILABLE(macos(10.3)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);
extern const CFStringRef kSCConsoleSessionUserName		/* value is CFString */
		API_AVAILABLE(macos(10.3)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);
extern const CFStringRef kSCConsoleSessionUID			/* value is CFNumber (a uid_t) */
		API_AVAILABLE(macos(10.3)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);
extern const CFStringRef kSCConsoleSessionConsoleSet		/* value is CFNumber */
		API_AVAILABLE(macos(10.3)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);
extern const CFStringRef kSCConsoleSessionOnConsole		/* value is CFBoolean */
		API_AVAILABLE(macos(10.3)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);
extern const CFStringRef kSCConsoleSessionLoginDone		/* value is CFBoolean */
		API_AVAILABLE(macos(10.3)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);
extern const CFStringRef kSCConsoleSessionSystemSafeBoot	/* value is CFBoolean */
		API_AVAILABLE(macos(10.3)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);
extern const CFStringRef kSCConsoleSessionLoginwindowSafeLogin	/* value is CFBoolean */
		API_AVAILABLE(macos(10.3)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);

/*!
	@function SCDynamicStoreCopyConsoleInformation
	@discussion Returns information about all console sessions on the system.
	@param store An SCDynamicStoreRef that should be used for communication
		with the server.
		If NULL, a temporary session will be used.
	@result An array of dictionaries containing information about each
		console session on the system; NULL if no sessions are defined
		or if an error was encountered.

		The contents of the returned array match that of the CoreGraphics
		CGSCopySessionList() SPI.

		You must release the returned value.
 */
CFArrayRef
SCDynamicStoreCopyConsoleInformation	(
					SCDynamicStoreRef	store
					)			API_AVAILABLE(macos(10.3)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);


__END_DECLS

#endif	/* _SCDYNAMICSTORECOPYSPECIFICPRIVATE_H */
