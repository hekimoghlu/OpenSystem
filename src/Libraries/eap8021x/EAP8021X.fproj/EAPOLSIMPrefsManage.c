/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
 *  EAPOLSIMPrefsManage.c
 * - routines to manage preferences for SIM 
 * - genration id is stored in System Preferences so eapolclient can know whether
 *   to use the information or not.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>
#include <TargetConditionals.h>
#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFPreferences.h>
#include <SystemConfiguration/SCValidation.h>
#include "symbol_scope.h"
#include "myCFUtil.h"
#include "EAPLog.h"
#include "EAPOLSIMPrefsManage.h"

#define kEAPOLSIMPrefsManageID		CFSTR("com.apple.eapol.sim.generation.plist")
#define kEAPOLSIMPrefsProcName		CFSTR("EAPOLSIMPrefsManage")
#define kEAPOLSIMPrefsGenIDKey		CFSTR("SIMGenerationID")

void
EAPOLSIMGenerationIncrement(void) {
    SCPreferencesRef	prefs = NULL;
    CFNumberRef		num = NULL;
    uint32_t 		value = 1;

    prefs = SCPreferencesCreate(NULL, kEAPOLSIMPrefsProcName, kEAPOLSIMPrefsManageID);
    if (prefs == NULL) {
	EAPLOG(LOG_NOTICE,
	       "SCPreferencesCreate failed, %s",
	       SCErrorString(SCError()));
	return;
    }

    num = SCPreferencesGetValue(prefs, kEAPOLSIMPrefsGenIDKey);
    num = isA_CFNumber(num);
    if (num != NULL) {
	CFNumberGetValue(num, kCFNumberSInt32Type, &value);
	++value;
    } 
    num = CFNumberCreate(NULL, kCFNumberSInt32Type, &value);	
    SCPreferencesSetValue(prefs, kEAPOLSIMPrefsGenIDKey, num);
    SCPreferencesCommitChanges(prefs);
    my_CFRelease(&num);
    my_CFRelease(&prefs);
    return;
}

UInt32
EAPOLSIMGenerationGet(void) {
    uint32_t 		ret_value = 0;
    SCPreferencesRef	prefs = NULL;
    CFNumberRef		num = NULL;

    prefs = SCPreferencesCreate(NULL, kEAPOLSIMPrefsProcName, kEAPOLSIMPrefsManageID);
    if (prefs == NULL) {
	EAPLOG(LOG_NOTICE,
	       "SCPreferencesCreate failed, %s",
	       SCErrorString(SCError()));
	       return 0;
    }
    num = SCPreferencesGetValue(prefs, kEAPOLSIMPrefsGenIDKey);
    num = isA_CFNumber(num);
    if (num == NULL) {
	goto done;;
    }
    CFNumberGetValue(num, kCFNumberSInt32Type, &ret_value);

done:
    my_CFRelease(&prefs);
    return ret_value;
}







