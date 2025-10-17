/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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
#ifndef __EAP8021X_EAPSIMAKAPERSISTENTSTATE_H__
#define __EAP8021X_EAPSIMAKAPERSISTENTSTATE_H__

#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFDate.h>
#include "EAPSIMAKA.h"

/* 
 * Modification History
 *
 * October 19, 2012	Dieter Siegmund (dieter@apple)
 * - created
 */

typedef struct EAPSIMAKAPersistentState *	EAPSIMAKAPersistentStateRef;

uint8_t *
EAPSIMAKAPersistentStateGetMasterKey(EAPSIMAKAPersistentStateRef persist);

int
EAPSIMAKAPersistentStateGetMasterKeySize(EAPSIMAKAPersistentStateRef persist);

CFStringRef
EAPSIMAKAPersistentStateGetIMSI(EAPSIMAKAPersistentStateRef persist);

CFStringRef
EAPSIMAKAPersistentStateGetPseudonym(EAPSIMAKAPersistentStateRef persist, 
				     CFDateRef * start_time);

Boolean
EAPSIMAKAPersistentStateTemporaryUsernameAvailable(EAPSIMAKAPersistentStateRef persist);

void
EAPSIMAKAPersistentStatePurgeTemporaryIDs(EAPSIMAKAPersistentStateRef persist);

CFStringRef
EAPSIMAKAPersistentStateGetSSID(EAPSIMAKAPersistentStateRef persist);

void
EAPSIMAKAPersistentStateSetPseudonym(EAPSIMAKAPersistentStateRef persist,
				     CFStringRef pseudonym);

CFStringRef
EAPSIMAKAPersistentStateGetReauthID(EAPSIMAKAPersistentStateRef persist);

Boolean
EAPSIMAKAPersistentStateGetReauthIDUsed(EAPSIMAKAPersistentStateRef persist);

void
EAPSIMAKAPersistentStateSetReauthID(EAPSIMAKAPersistentStateRef persist,
				    CFStringRef reauth_id);

void
EAPSIMAKAPersistentStateSetReauthIDUsed(EAPSIMAKAPersistentStateRef persist,
					  Boolean reauth_id_used);

uint16_t
EAPSIMAKAPersistentStateGetCounter(EAPSIMAKAPersistentStateRef persist);

void
EAPSIMAKAPersistentStateSetCounter(EAPSIMAKAPersistentStateRef persist,
				   uint16_t counter);
EAPSIMAKAPersistentStateRef
EAPSIMAKAPersistentStateCreate(EAPType type, int master_key_size,
			       CFStringRef imsi,
			       EAPSIMAKAAttributeType identity_type,
			       CFStringRef ssid);
void
EAPSIMAKAPersistentStateSave(EAPSIMAKAPersistentStateRef persist,
			     Boolean master_key_valid);
void
EAPSIMAKAPersistentStateRelease(EAPSIMAKAPersistentStateRef persist);

void
EAPSIMAKAPersistentStateForgetSSID(CFStringRef ssid);

#endif /* __EAP8021X_EAPSIMAKAPERSISTENTSTATE_H__ */
