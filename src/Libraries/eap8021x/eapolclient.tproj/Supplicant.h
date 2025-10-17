/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
 * November 8, 2001	Dieter Siegmund (dieter@apple.com)
 * - created
 */

#ifndef _S_SUPPLICANT_H
#define _S_SUPPLICANT_H

#include <CoreFoundation/CFDictionary.h>
#include <sys/types.h>
#include <net/if_dl.h>
#include "EAPClientTypes.h"
#include "SupplicantTypes.h"
#include "EAPOLControlTypes.h"
#include "EAPOLSocket.h"

#define BAD_IDENTIFIER		(-1)

typedef struct Supplicant_s Supplicant, *SupplicantRef;

SupplicantRef 
Supplicant_create(EAPOLSocketRef sock);

SupplicantRef
Supplicant_create_with_supplicant(EAPOLSocketRef sock, SupplicantRef main_supp);

void
Supplicant_free(SupplicantRef * supp_p);

void
Supplicant_start(SupplicantRef supp, int packet_identifier);

void
Supplicant_set_no_ui(SupplicantRef supp);

void
Supplicant_link_status_changed(SupplicantRef supp, bool active);

SupplicantState
Supplicant_get_state(SupplicantRef supp, EAPClientStatus * last_status);

bool
Supplicant_control(SupplicantRef supp,
		   EAPOLClientControlCommand command,
		   CFDictionaryRef control_dict);

bool
Supplicant_update_configuration(SupplicantRef supp,
				CFDictionaryRef config_dict,
				bool * should_stop);

void
Supplicant_stop(SupplicantRef supp);

void
Supplicant_simulate_success(SupplicantRef supp);

void
Supplicant_set_globals(SCPreferencesRef prefs);

#endif /* _S_SUPPLICANT_H */

