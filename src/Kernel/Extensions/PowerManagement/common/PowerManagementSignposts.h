/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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

//
//  PowerManagementSignposts.h
//  PowerManagement
//
//  Created by John Scheible on 5/11/18.
//

#ifndef PowerManagementSignposts_h
#define PowerManagementSignposts_h

#include <sys/kdebug.h>

#define SYSTEMCHARGING_ARIADNE_SUBCLASS  103
#define PROJECT_SHIFT 8
#define PROJECT_CODE 0x02
#define SYSTEMCHARGINGDBG_CODE(event) ARIADNEDBG_CODE(SYSTEMCHARGING_ARIADNE_SUBCLASS, (PROJECT_CODE << PROJECT_SHIFT) | event)

// AppleSmartBatteryManager
#define SYSTEMCHARGING_ASBM_READ_EXTERNAL_CONNECTED       SYSTEMCHARGINGDBG_CODE(0x00) // Impulse
#define SYSTEMCHARGING_ASBM_BATTERY_POLL                  SYSTEMCHARGINGDBG_CODE(0x01) // Interval
#define SYSTEMCHARGING_ASBM_SMC_KEY_READ                  SYSTEMCHARGINGDBG_CODE(0x02) // Interval
#define SYSTEMCHARGING_ASBM_SMC_KEY_WRITE                 SYSTEMCHARGINGDBG_CODE(0x03) // Interval
#define SYSTEMCHARGING_ASBM_SMC_KEY_INFO                  SYSTEMCHARGINGDBG_CODE(0x04) // Interval
#define SYSTEMCHARGING_ASBM_UPDATE_POWER_SOURCE           SYSTEMCHARGINGDBG_CODE(0x05) // Impulse
// powerd
#define SYSTEMCHARGING_POWERD_HANDLE_BATTERY_STATUS_UPDATE      SYSTEMCHARGINGDBG_CODE(0x80) // Impulse
#define SYSTEMCHARGING_POWERD_PUBLISH_POWER_SOURCE_CHANGE       SYSTEMCHARGINGDBG_CODE(0x81) // Impulse



#endif /* PowerManagementSignposts_h */
