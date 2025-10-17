/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 12, 2024.
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
#include "SCHelper_client.h"

#include <unistd.h>
#include <pthread.h>


static Boolean
__SCPreferencesUnlock_helper(SCPreferencesRef prefs)
{
	Boolean			ok;
	SCPreferencesPrivateRef	prefsPrivate	= (SCPreferencesPrivateRef)prefs;
	uint32_t		status		= kSCStatusOK;

	if (prefsPrivate->helper_port == MACH_PORT_NULL) {
		// if no helper
		goto fail;
	}

	// have the helper "unlock" the prefs
//	status = kSCStatusOK;
	ok = _SCHelperExec(prefsPrivate->helper_port,
			   SCHELPER_MSG_PREFS_UNLOCK,
			   NULL,
			   &status,
			   NULL);
	if (!ok) {
		goto fail;
	}

	if (status != kSCStatusOK) {
		goto error;
	}

	__SCPreferencesUpdateLockedState(prefs, FALSE);
	return TRUE;

    fail :

	// close helper
	if (prefsPrivate->helper_port != MACH_PORT_NULL) {
		_SCHelperClose(&prefsPrivate->helper_port);
	}

	status = kSCStatusAccessError;

    error :

	// return error
	_SCErrorSet(status);
	return FALSE;
}


static void
reportDelay(SCPreferencesRef prefs, struct timeval *delay)
{
	SCPreferencesPrivateRef	prefsPrivate	= (SCPreferencesPrivateRef)prefs;

	SC_log(LOG_ERR,
	       "SCPreferences(%@:%@) lock held for %d.%3.3d seconds",
	       prefsPrivate->name,
	       prefsPrivate->prefsID,
	       (int)delay->tv_sec,
	       delay->tv_usec / 1000);
	return;
}


Boolean
SCPreferencesUnlock(SCPreferencesRef prefs)
{
	struct timeval		lockElapsed;
	struct timeval		lockEnd;
	SCPreferencesPrivateRef	prefsPrivate	= (SCPreferencesPrivateRef)prefs;

	if (prefs == NULL) {
		/* sorry, you must provide a session */
		_SCErrorSet(kSCStatusNoPrefsSession);
		return FALSE;
	}

	if (!prefsPrivate->locked) {
		/* sorry, you don't have the lock */
		_SCErrorSet(kSCStatusNeedLock);
		return FALSE;
	}

	if (prefsPrivate->authorizationData != NULL) {
		return __SCPreferencesUnlock_helper(prefs);
	}

	pthread_mutex_lock(&prefsPrivate->lock);

	if (prefsPrivate->sessionNoO_EXLOCK != NULL) {
		// Note: closing the session removes the temporary "lock" key
		CFRelease(prefsPrivate->sessionNoO_EXLOCK);
		prefsPrivate->sessionNoO_EXLOCK = NULL;
	}

	if (prefsPrivate->lockFD != -1)	{
		if (prefsPrivate->lockPath != NULL) {
			unlink(prefsPrivate->lockPath);
		}
		close(prefsPrivate->lockFD);
		prefsPrivate->lockFD = -1;
	}

	(void)gettimeofday(&lockEnd, NULL);
	timersub(&lockEnd, &prefsPrivate->lockTime, &lockElapsed);
	if (lockElapsed.tv_sec > 0) {
		// if we held the lock for more than 1 second
		reportDelay(prefs, &lockElapsed);
	}

	SC_log(LOG_DEBUG, "SCPreferences() unlock: %s",
	       prefsPrivate->path);

	__SCPreferencesUpdateLockedState(prefs, FALSE);

	pthread_mutex_unlock(&prefsPrivate->lock);
	return TRUE;
}
