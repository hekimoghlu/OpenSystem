/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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
 * Nov 28, 2001			Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#include <TargetConditionals.h>
#include <SystemConfiguration/SystemConfiguration.h>
#include <SystemConfiguration/SCValidation.h>
#include <SystemConfiguration/SCPrivate.h>

CFStringRef
SCDynamicStoreKeyCreateLocation(CFAllocatorRef allocator)
{
#pragma unused(allocator)
	return CFRetain(kSCDynamicStoreDomainSetup);
}


#if	!TARGET_OS_IPHONE
CFStringRef
SCDynamicStoreCopyLocation(SCDynamicStoreRef store)
{
	CFDictionaryRef		dict		= NULL;
	CFStringRef		key;
	CFStringRef		location	= NULL;

	key  = SCDynamicStoreKeyCreateLocation(NULL);
	dict = SCDynamicStoreCopyValue(store, key);
	CFRelease(key);
	if (!isA_CFDictionary(dict)) {
		_SCErrorSet(kSCStatusNoKey);
		goto done;
	}

	location = CFDictionaryGetValue(dict, kSCDynamicStorePropSetupCurrentSet);
	location = isA_CFString(location);
	if (location == NULL) {
		_SCErrorSet(kSCStatusNoKey);
		goto done;
	}

	CFRetain(location);

    done :

	if (dict)		CFRelease(dict);

	return location;
}
#endif	// !TARGET_OS_IPHONE
