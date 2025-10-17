/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 8, 2023.
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
 * May 21, 2008	Dieter Siegmund (dieter@apple.com)
 * - created
 */

#ifndef _S_EAPOLSOCKETPRIVATE_H
#define _S_EAPOLSOCKETPRIVATE_H
#include <stdio.h>
#include <net/ethernet.h>
#include <SystemConfiguration/SCPreferences.h>
#include "Supplicant.h"

typedef struct EAPOLSocketSource_s EAPOLSocketSource, *EAPOLSocketSourceRef;

EAPOLSocketSourceRef
EAPOLSocketSourceCreate(const char * if_name,
			const struct ether_addr * ether,
			CFDictionaryRef * control_dict_p);
void
EAPOLSocketSourceFree(EAPOLSocketSourceRef * source_p);

SupplicantRef
EAPOLSocketSourceCreateSupplicant(EAPOLSocketSourceRef source,
				  CFDictionaryRef control_dict,
				  int * packet_identifier);
void
EAPOLSocketSetGlobals(SCPreferencesRef prefs);

#endif /* _S_EAPOLSOCKETPRIVATE_H */

