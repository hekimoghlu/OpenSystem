/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
 * September 8, 2016	Allan Nathanson <ajn@apple.com>
 * - initial revision
 */


#ifndef	_PLUGIN_SHARED_H
#define	_PLUGIN_SHARED_H


#include <CoreFoundation/CoreFoundation.h>


#pragma mark -
#pragma mark InterfaceNamer.bundle --> others

/*
 * Plugin:InterfaceNamer [SCDynamicStore] dictionary content
 */

// IORegistry "quiet", "complete" (last boot interfaces found), and "timeout"
#define	kInterfaceNamerKey_Complete			CFSTR("*COMPLETE*")
#define	kInterfaceNamerKey_Quiet			CFSTR("*QUIET*")
#define	kInterfaceNamerKey_Timeout			CFSTR("*TIMEOUT*")

// Configuration excluded network interfaces
#define	kInterfaceNamerKey_ExcludedInterfaces		CFSTR("_Excluded_")

// Network interfaces that have not yet been made available because the console is "locked"
#define	kInterfaceNamerKey_LockedInterfaces		CFSTR("_Locked_")

// [Apple] pre-configured network interfaces
#define	kInterfaceNamerKey_PreConfiguredInterfaces	CFSTR("_PreConfigured_")

// BT-PAN network interfaces
#define	BT_PAN_NAME					"Bluetooth PAN"
#define	kInterfaceNamerKey_BT_PAN_Name			CFSTR("_" BT_PAN_NAME "_")
#define	kInterfaceNamerKey_BT_PAN_Mac			CFSTR("_" BT_PAN_NAME " (MAC)" "_")


#endif	/* _PLUGIN_SHARED_H */
