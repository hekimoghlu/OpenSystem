/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 22, 2023.
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
#include <err.h>
#include "preferences.h"
#include "plog.h"

SCPreferencesRef				gPrefs = NULL;

static SCPreferencesContext		prefsContext = { 0, NULL, NULL, NULL, NULL };

static void
prefscallout (SCPreferencesRef           prefs,
			  SCPreferencesNotification  notificationType,
			  void                      *context)
{
	if ((notificationType & kSCPreferencesNotificationApply) != 0) {
		// other prefs here
		plogreadprefs();
	}
	
	return;
}

void
prefsinit (void)
{
	if (!gPrefs) {
		if ((gPrefs = SCPreferencesCreate(0, CFSTR("racoon"), CFSTR("com.apple.ipsec.plist")))) {
			if (SCPreferencesSetCallback(gPrefs, prefscallout, &prefsContext)) {
				if (!SCPreferencesSetDispatchQueue(gPrefs, dispatch_get_main_queue())) {
					errx(1, "failed to initialize dispatch queue.\n");
				}
			}
		}
	}
}

