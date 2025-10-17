/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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


static Boolean
__SCPreferencesApplyChanges_helper(SCPreferencesRef prefs)
{
	Boolean			ok;
	SCPreferencesPrivateRef	prefsPrivate	= (SCPreferencesPrivateRef)prefs;
	uint32_t		status		= kSCStatusOK;

	if (prefsPrivate->helper_port == MACH_PORT_NULL) {
		// if no helper
		goto fail;
	}

	// have the helper "apply" the prefs
//	status = kSCStatusOK;
	ok = _SCHelperExec(prefsPrivate->helper_port,
			   SCHELPER_MSG_PREFS_APPLY,
			   NULL,
			   &status,
			   NULL);
	if (!ok) {
		goto fail;
	}

	if (status != kSCStatusOK) {
		goto error;
	}

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


Boolean
SCPreferencesApplyChanges(SCPreferencesRef prefs)
{
	Boolean			ok		= FALSE;
	SCPreferencesPrivateRef	prefsPrivate	= (SCPreferencesPrivateRef)prefs;
	Boolean			wasLocked;

	if (prefs == NULL) {
		/* sorry, you must provide a session */
		_SCErrorSet(kSCStatusNoPrefsSession);
		return FALSE;
	}

	/*
	 * Determine if the we have exclusive access to the preferences
	 * and acquire the lock if necessary.
	 */
	wasLocked = prefsPrivate->locked;
	if (!wasLocked) {
		if (!SCPreferencesLock(prefs, TRUE)) {
			SC_log(LOG_INFO, "SCPreferencesLock() failed");
			return FALSE;
		}
	}

	if (prefsPrivate->authorizationData != NULL) {
		ok = __SCPreferencesApplyChanges_helper(prefs);
		goto done;
	}

	SC_log(LOG_INFO, "SCPreferences() apply: %s",
	       prefsPrivate->path);

	/* post notification */
	ok = SCDynamicStoreNotifyValue(NULL, prefsPrivate->sessionKeyApply);
	if (!ok) {
		SC_log(LOG_INFO, "SCDynamicStoreNotifyValue() failed");
		_SCErrorSet(kSCStatusFailed);
	}

    done :

	if (!wasLocked) {
		uint32_t	status;

		status = SCError();	// preserve status across unlock
		(void) SCPreferencesUnlock(prefs);
		_SCErrorSet(status);
	}
	return ok;
}
