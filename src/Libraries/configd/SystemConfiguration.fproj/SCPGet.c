/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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
/*
 * Modification History
 *
 * June 1, 2001			Allan Nathanson <ajn@apple.com>
 * - public API conversion
 *
 * November 9, 2000		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#include <fcntl.h>
#include <unistd.h>
#include <sys/errno.h>

#include "SCPreferencesInternal.h"

CFPropertyListRef
SCPreferencesGetValue(SCPreferencesRef prefs, CFStringRef key)
{
	SCPreferencesPrivateRef	prefsPrivate	= (SCPreferencesPrivateRef)prefs;
	CFPropertyListRef	value;

	if (prefs == NULL) {
		/* sorry, you must provide a session */
		_SCErrorSet(kSCStatusNoPrefsSession);
		return NULL;
	}

	__SCPreferencesAccess(prefs);

	value = CFDictionaryGetValue(prefsPrivate->prefs, key);
	if (value == NULL) {
		_SCErrorSet(kSCStatusNoKey);
	}

	return value;
}
