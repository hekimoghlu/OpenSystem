/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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
/******************************************************************************
	event_status_driver.h
	API for the events status driver.
	This file contains public API.
	mpaque 11Oct91
	
	Copyright 1991 NeXT Computer, Inc.
	
	Modified:
	
******************************************************************************/

#ifndef _DRIVERS_EVENT_STATUS_DRIVER_
#define _DRIVERS_EVENT_STATUS_DRIVER_

__BEGIN_DECLS

#include <mach/port.h>
#include <IOKit/hidsystem/IOLLEvent.h>
#include <IOKit/hidsystem/IOHIDTypes.h>
#include <AvailabilityMacros.h> 
#include <IOKit/IOKitLib.h>

/*
 * Event System Handle:
 *
 * Information used by the system between calls to NXOpenEventSystem and
 * NXCloseEventSystem.  The application should not
 * access any of the elements of this structure.
 */
typedef mach_port_t NXEventHandle;

/* Open and Close */
NXEventHandle NXOpenEventStatus(void) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_12, __IPHONE_NA, __IPHONE_NA);
void NXCloseEventStatus(NXEventHandle handle) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_12, __IPHONE_NA, __IPHONE_NA);

/* Status */
extern NXEventSystemInfoType NXEventSystemInfo(NXEventHandle handle,
				char *flavor,
				int *evs_info,
				unsigned int *evs_info_cnt) __deprecated;
/* Keyboard */
extern void NXSetKeyRepeatInterval(NXEventHandle handle, double seconds) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_12, __IPHONE_NA, __IPHONE_NA);
extern double NXKeyRepeatInterval(NXEventHandle handle) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_12, __IPHONE_NA, __IPHONE_NA);
extern void NXSetKeyRepeatThreshold(NXEventHandle handle, double threshold) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_12, __IPHONE_NA, __IPHONE_NA);
extern double NXKeyRepeatThreshold(NXEventHandle handle) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_12, __IPHONE_NA, __IPHONE_NA);
extern void NXResetKeyboard(NXEventHandle handle) __deprecated;

/* Mouse */
extern void NXSetClickTime(NXEventHandle handle, double seconds)__OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_12, __IPHONE_NA, __IPHONE_NA);
extern double NXClickTime(NXEventHandle handle) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_12, __IPHONE_NA, __IPHONE_NA);
extern void NXSetClickSpace(NXEventHandle handle, _NXSize_ *area) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_12, __IPHONE_NA, __IPHONE_NA);
extern void NXGetClickSpace(NXEventHandle handle, _NXSize_ *area) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_12, __IPHONE_NA, __IPHONE_NA);
extern void NXResetMouse(NXEventHandle handle) __deprecated;

/*
* The following functions have been removed.
*
* NXIdleTime
* ÊÊÊÊSee CGEventSourceSecondsSinceLastEventType.
*
* NXSetKeyMapping
* NXKeyMappingLength
* NXGetKeyMapping
* ÊÊÊÊThese do not have a drop in replacement. ÊSee UCKeyTranslate.
*
* NXSetMouseScaling
* ÊÊÊÊSee IOHIDSetAccelerationWithKey and IOHIDSetMouseAcceleration.
*
* NXGetMouseScaling
* ÊÊÊÊSee IOHIDGetAccelerationWithKey and IOHIDGetMouseAcceleration.
*
* NXSetAutoDimThreshold
* NXSetAutoDimState
* ÊÊÊÊSee IOPMSetAggressiveness and kPMMinutesToDim.
*
* NXAutoDimThreshold
* NXAutoDimTime
* NXAutoDimState
* ÊÊÊÊSee IOPMGetAggressiveness and kPMMinutesToDim.
*
* NXSetAutoDimBrightness
* NXAutoDimBrightness
* NXSetScreenBrightness
* NXScreenBrightness
* ÊÊÊÊThis functionality is unsupported.
*/

__END_DECLS

#endif /*_DRIVERS_EVENT_STATUS_DRIVER_ */

