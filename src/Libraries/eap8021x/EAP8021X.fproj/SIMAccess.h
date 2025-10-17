/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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
#ifndef _EAP8021X_SIMACCESS_H
#define _EAP8021X_SIMACCESS_H


/* 
 * Modification History
 *
 * January 15, 2009	Dieter Siegmund (dieter@apple.com)
 * - created
 */

/*
 * SIMAccess.h
 * - API's to access the SIM
 */

#include <stdint.h>
#include <stdbool.h>
#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFRunLoop.h>
#include "EAPSIMAKA.h"

CFStringRef
SIMCopyIMSI(CFDictionaryRef properties);

CFStringRef
SIMCopyRealm(CFDictionaryRef properties);

CFDictionaryRef
SIMCopyEncryptedIMSIInfo(EAPType type);

Boolean
SIMIsOOBPseudonymSupported(Boolean *isSupported);

CFStringRef
SIMCopyOOBPseudonym();

void
SIMReportDecryptionError(CFDataRef encryptedIdentity);

/*
 * Function: SIMAuthenticateGSM
 * Purpose:
 *   Communicate with SIM to retrieve the (SRES, Kc) pairs for the given
 *   set of RANDs.
 * Parameters:
 *   rand_p		input buffer containing RANDs;
 *			size must be at least 'count' * SIM_RAND_SIZE
 *   count		the number of values in rand_p, kc_p, and sres_p
 *   kc_p		output buffer to return Kc values;
 *			size must be at least 'count' * SIM_KC_SIZE
 *   sres_p		output buffer to return SRES values;
 * 			size must be at least 'count' * SIM_SRES_SIZE
 * Returns:
 *   TRUE if RANDS were processed and kc_p and sres_p were filled in,
 *   FALSE on failure.
 */
bool
SIMAuthenticateGSM(CFDictionaryRef properties, const uint8_t * rand_p, int count,
		   uint8_t * kc_p, uint8_t * sres_p);

typedef struct {
    CFDataRef	ck;
    CFDataRef	ik;
    CFDataRef	res;
    CFDataRef	auts;
} AKAAuthResults, * AKAAuthResultsRef;

void
AKAAuthResultsSetCK(AKAAuthResultsRef results, CFDataRef ck);

void
AKAAuthResultsSetIK(AKAAuthResultsRef results, CFDataRef ik);

void
AKAAuthResultsSetRES(AKAAuthResultsRef results, CFDataRef res);

void
AKAAuthResultsSetAUTS(AKAAuthResultsRef results, CFDataRef auts);

void
AKAAuthResultsInit(AKAAuthResultsRef results);

void
AKAAuthResultsRelease(AKAAuthResultsRef results);

/*
 * Function: SIMAuthenticateAKA
 * Purpose:
 *   Run the AKA algorithms on the AT_RAND data.
 *
 * Returns:
 *   FALSE if the request could not be completed (SIM unavailable).
 *
 *   TRUE if results are available:
 *   - if authentication was successful, AKAAuthResultsRef contains non-NULL
 *     res, ck, and ik values.
 *   - if there's a sync failure, AKAAuthResultsRef will contain non-NULL
 *     auts value.
 *   - otherwise, there was an auth reject.
 */
bool
SIMAuthenticateAKA(CFDictionaryRef properties, CFDataRef rand, CFDataRef autn, AKAAuthResultsRef results);

#endif /* _EAP8021X_SIMACCESS_H */

