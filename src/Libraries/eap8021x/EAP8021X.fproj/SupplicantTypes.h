/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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
#ifndef _EAP8021X_SUPPLICANTTYPES_H
#define _EAP8021X_SUPPLICANTTYPES_H

#include <stdint.h>

enum {
    kSupplicantStateDisconnected = 0,
    kSupplicantStateConnecting = 1,
    kSupplicantStateAcquired = 2,
    kSupplicantStateAuthenticating = 3,
    kSupplicantStateAuthenticated = 4,
    kSupplicantStateHeld = 5,
    kSupplicantStateLogoff = 6,
    kSupplicantStateInactive = 7,
    kSupplicantStateNoAuthenticator = 8,
    kSupplicantStateFirst = kSupplicantStateDisconnected,
    kSupplicantStateLast = kSupplicantStateNoAuthenticator
};

typedef uint32_t SupplicantState;
						
static __inline__ const char *
SupplicantStateString(SupplicantState state)
{
    static const char * str[] = {
	"Disconnected",
	"Connecting",
	"Acquired",
	"Authenticating",
	"Authenticated",
	"Held",
	"Logoff",
	"Inactive",
	"No Authenticator"
    };

    if (state >= kSupplicantStateFirst
	&& state <= kSupplicantStateLast) {
	return (str[state]);
    }
    return ("<unknown>");
}

#endif /* _EAP8021X_SUPPLICANTTYPES_H */
