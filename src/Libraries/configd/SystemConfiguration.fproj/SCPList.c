/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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

#include "SCPreferencesInternal.h"

CFArrayRef
SCPreferencesCopyKeyList(SCPreferencesRef prefs)
{
	CFAllocatorRef		allocator;
	CFArrayRef		keys;
	SCPreferencesPrivateRef	prefsPrivate	= (SCPreferencesPrivateRef)prefs;
	CFIndex			prefsCnt;
	const void **		prefsKeys;

	if (prefs == NULL) {
		/* sorry, you must provide a session */
		_SCErrorSet(kSCStatusNoPrefsSession);
		return NULL;
	}

	__SCPreferencesAccess(prefs);

	allocator = CFGetAllocator(prefs);
	prefsCnt  = CFDictionaryGetCount(prefsPrivate->prefs);
	if (prefsCnt > 0) {
		prefsKeys = CFAllocatorAllocate(allocator, prefsCnt * sizeof(CFStringRef), 0);
		CFDictionaryGetKeysAndValues(prefsPrivate->prefs, prefsKeys, NULL);
		keys = CFArrayCreate(allocator, prefsKeys, prefsCnt, &kCFTypeArrayCallBacks);
		CFAllocatorDeallocate(allocator, prefsKeys);
	} else {
		keys = CFArrayCreate(allocator, NULL, 0, &kCFTypeArrayCallBacks);
	}

	return keys;
}
