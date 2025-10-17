/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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

#include "scutil.h"
#include "cache.h"
#include "session.h"
#include "notifications.h"


static void
reconnected(SCDynamicStoreRef store, void *info)
{
#pragma unused(store)
#pragma unused(info)
	SCPrint(TRUE, stdout, CFSTR("SCDynamicStore server restarted, session reconnected\n"));
	return;
}


__private_extern__
void
do_open(int argc, char * const argv[])
{
#pragma unused(argv)
	if (store) {
		CFRelease(store);
		CFRelease(watchedKeys);
		CFRelease(watchedPatterns);
	}

	if (argc < 1) {
		store = SCDynamicStoreCreate(NULL, CFSTR("scutil"), storeCallback, NULL);
	} else {
		CFMutableDictionaryRef	options;

		options = CFDictionaryCreateMutable(NULL,
						    0,
						    &kCFTypeDictionaryKeyCallBacks,
						    &kCFTypeDictionaryValueCallBacks);
		CFDictionarySetValue(options, kSCDynamicStoreUseSessionKeys, kCFBooleanTrue);
		store = SCDynamicStoreCreateWithOptions(NULL,
							CFSTR("scutil"),
							options,
							storeCallback,
							NULL);
		CFRelease(options);
	}
	if (store == NULL) {
		SCPrint(TRUE, stdout, CFSTR("  %s\n"), SCErrorString(SCError()));
		return;
	}

	(void) SCDynamicStoreSetDisconnectCallBack(store, reconnected);

	watchedKeys     = CFArrayCreateMutable(NULL, 0, &kCFTypeArrayCallBacks);
	watchedPatterns = CFArrayCreateMutable(NULL, 0, &kCFTypeArrayCallBacks);

	_SCDynamicStoreCacheClose(store);

	return;
}


__private_extern__
void
do_close(int argc, char * const argv[])
{
#pragma unused(argc)
#pragma unused(argv)
	if (notifyRls != NULL) {
		if (doDispatch) {
			(void) SCDynamicStoreSetDispatchQueue(store, NULL);
		} else {
			CFRunLoopSourceInvalidate(notifyRls);
			CFRelease(notifyRls);
		}
		notifyRls = NULL;
	}

	if (notifyRl != NULL) {
		CFRunLoopStop(notifyRl);
		notifyRl  = NULL;
	}

	if (store != NULL) {
		CFRelease(store);
		store = NULL;
		CFRelease(watchedKeys);
		watchedKeys = NULL;
		CFRelease(watchedPatterns);
		watchedPatterns = NULL;
	}

	_SCDynamicStoreCacheClose(store);

	return;
}
