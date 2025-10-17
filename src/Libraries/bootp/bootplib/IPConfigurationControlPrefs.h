/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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
#ifndef _S_IPCONFIGURATIONCONTROLPREFS_H
#define _S_IPCONFIGURATIONCONTROLPREFS_H

/*
 * IPConfigurationControlPrefs.h
 * - definitions for accessing IPConfiguration controlpreferences and being
 *   notified when they change
 */

/* 
 * Modification History
 *
 * March 26, 2013	Dieter Siegmund (dieter@apple)
 * - created (from EAPOLControlPrefs.h)
 */
#include <SystemConfiguration/SCPreferences.h>

#include "DHCPDUID.h"

typedef void (*IPConfigurationControlPrefsCallBack)(SCPreferencesRef prefs);

SCPreferencesRef
IPConfigurationControlPrefsInit(dispatch_queue_t queue,
				IPConfigurationControlPrefsCallBack callback);

void
IPConfigurationControlPrefsSynchronize(void);

Boolean
IPConfigurationControlPrefsGetVerbose(Boolean default_val);

Boolean
IPConfigurationControlPrefsSetVerbose(Boolean verbose);

typedef CF_ENUM(uint32_t, IPConfigurationInterfaceTypes) {
    kIPConfigurationInterfaceTypesUnspecified = 0,
    kIPConfigurationInterfaceTypesNone = 1,
    kIPConfigurationInterfaceTypesCellular = 2,
    kIPConfigurationInterfaceTypesAll = 3,
};

IPConfigurationInterfaceTypes
IPConfigurationInterfaceTypesFromString(CFStringRef str);

CFStringRef
IPConfigurationInterfaceTypesToString(IPConfigurationInterfaceTypes types);

Boolean
IPConfigurationControlPrefsSetAWDReportInterfaceTypes(IPConfigurationInterfaceTypes
						      types);

IPConfigurationInterfaceTypes
IPConfigurationControlPrefsGetAWDReportInterfaceTypes(void);

Boolean
IPConfigurationControlPrefsGetCellularCLAT46AutoEnable(Boolean default_val);

Boolean
IPConfigurationControlPrefsSetCellularCLAT46AutoEnable(Boolean enable);

Boolean
IPConfigurationControlPrefsGetIPv6LinkLocalModifierExpires(Boolean default_val);

Boolean
IPConfigurationControlPrefsSetIPv6LinkLocalModifierExpires(Boolean expires);

DHCPDUIDType
IPConfigurationControlPrefsGetDHCPDUIDType(void);

Boolean
IPConfigurationControlPrefsSetDHCPDUIDType(DHCPDUIDType type);

Boolean
IPConfigurationControlPrefsGetHideBSSID(Boolean default_val,
                                        Boolean * ret_was_set);
Boolean
IPConfigurationControlPrefsSetHideBSSID(Boolean hide);

Boolean
IPConfigurationControlPrefsSetHideBSSIDDefault(void);

#endif /* _S_IPCONFIGURATIONCONTROLPREFS_H */
