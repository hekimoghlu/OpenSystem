/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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
 * Oct 1, 2018	Allan Nathanson <ajn@apple.com>
 * - initial revision
 */


#ifdef	SC_LOG_HANDLE
#include <os/log.h>
os_log_t	SC_LOG_HANDLE(void);
#endif	//SC_LOG_HANDLE

#include "SCDynamicStoreInternal.h"


Boolean
_SCDynamicStoreCacheIsActive(SCDynamicStoreRef store)
{
	SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)store;

	if (!isA_SCDynamicStore(store)) {
		// sorry, you must provide a session
		_SCErrorSet(kSCStatusNoStoreSession);
		return FALSE;
	}

	return storePrivate->cache_active;
}


static void
__SCDynamicStoreCacheRelease(SCDynamicStoreRef store)
{
	SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)store;

	if (storePrivate->cached_keys != NULL) {
		CFRelease(storePrivate->cached_keys);
		storePrivate->cached_keys = NULL;
	}
	if (storePrivate->cached_set != NULL) {
		CFRelease(storePrivate->cached_set);
		storePrivate->cached_set = NULL;
	}
	if (storePrivate->cached_removals != NULL) {
		CFRelease(storePrivate->cached_removals);
		storePrivate->cached_removals = NULL;
	}
	if (storePrivate->cached_notifys != NULL) {
		CFRelease(storePrivate->cached_notifys);
		storePrivate->cached_notifys = NULL;
	}

	return;
}


Boolean
_SCDynamicStoreCacheOpen(SCDynamicStoreRef store)
{
	SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)store;

	if (!isA_SCDynamicStore(store)) {
		// sorry, you must provide a session
		_SCErrorSet(kSCStatusNoStoreSession);
		return FALSE;
	}

	__SCDynamicStoreCacheRelease(store);	// if we are already using the cache, start clean
	storePrivate->cache_active = TRUE;

	return TRUE;
}


Boolean
_SCDynamicStoreCacheCommitChanges(SCDynamicStoreRef store)
{
	Boolean				ok		= TRUE;
	SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)store;

	if (!isA_SCDynamicStore(store)) {
		// sorry, you must provide a session
		_SCErrorSet(kSCStatusNoStoreSession);
		return FALSE;
	}

	if (!storePrivate->cache_active) {
		// if not using the cache
		_SCErrorSet(kSCStatusFailed);
		return FALSE;
	}

	if ((storePrivate->cached_set != NULL) ||
	    (storePrivate->cached_removals != NULL) ||
	    (storePrivate->cached_notifys != NULL)) {
		ok = SCDynamicStoreSetMultiple(store,
					       storePrivate->cached_set,
					       storePrivate->cached_removals,
					       storePrivate->cached_notifys);
		__SCDynamicStoreCacheRelease(store);
	}

	return ok;
}


Boolean
_SCDynamicStoreCacheClose(SCDynamicStoreRef store)
{
	SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)store;

	if (!isA_SCDynamicStore(store)) {
		// sorry, you must provide a session
		_SCErrorSet(kSCStatusNoStoreSession);
		return FALSE;
	}

	if (!storePrivate->cache_active) {
		// if not using the cache
		_SCErrorSet(kSCStatusFailed);
		return FALSE;
	}

	__SCDynamicStoreCacheRelease(store);
	storePrivate->cache_active = FALSE;

	return TRUE;
}
