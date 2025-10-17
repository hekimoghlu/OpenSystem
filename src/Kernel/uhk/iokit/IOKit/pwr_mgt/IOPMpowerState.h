/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
#ifndef _IOKIT_IOPMPOWERSTATE_H
#define _IOKIT_IOPMPOWERSTATE_H

#include <IOKit/pwr_mgt/IOPM.h>

/*! @header IOPMpowerState.h
 *   @abstract Defines the struct IOPMPowerState that power managed drivers should use to describe their power states.
 */

/*! @struct IOPMPowerState
 *   @abstract Describes a device's power state.
 *   @discussion To take part in system power management, drivers should define an array of 2 or more power states and pass it to kernel power management through IOService::registerPowerDriver.
 *   @field version Defines version number of this struct. Just use the value "1" when defining an IOPMPowerState.
 *   @field	capabilityFlags Describes the capability of the device in this state.
 *   @field	outputPowerCharacter Describes the power provided in this state.
 *   @field	inputPowerRequirement Describes the input power required in this state.
 *   @field	staticPower Describes average consumption in milliwatts. Unused; drivers may specify 0.
 *   @field	stateOrder Valid in version kIOPMPowerStateVersion2 or greater of this structure. Defines ordering of power states independently of the power state ordinal.
 *   @field	powerToAttain Describes dditional power to attain this state from next lower state (in milliWatts). Unused; drivers may specify 0.
 *   @field	timeToAttain Describes time required to enter this state from next lower state (in microseconds). Unused; drivers may specify 0.
 *   @field	settleUpTime Describes settle time required after entering this state from next lower state (microseconds). Unused; drivers may specify 0.
 *   @field timeToLower Describes time required to enter next lower state from this one (microseconds). Unused; drivers may specify 0.
 *   @field	settleDownTime Settle time required after entering next lower state from this state (microseconds). Unused; drivers may specify 0.
 *   @field	powerDomainBudget Describes power in milliWatts a domain in this state can deliver to its children. Unused; drivers may specify 0.
 *  }
 */

struct IOPMPowerState {
	unsigned long       version;
	IOPMPowerFlags      capabilityFlags;
	IOPMPowerFlags      outputPowerCharacter;
	IOPMPowerFlags      inputPowerRequirement;
	unsigned long       staticPower;
	unsigned long       stateOrder;
	unsigned long       powerToAttain;
	unsigned long       timeToAttain;
	unsigned long       settleUpTime;
	unsigned long       timeToLower;
	unsigned long       settleDownTime;
	unsigned long       powerDomainBudget;
};

typedef struct IOPMPowerState IOPMPowerState;

enum {
	kIOPMPowerStateVersion1 = 1,
	kIOPMPowerStateVersion2 = 2
};

#endif /* _IOKIT_IOPMPOWERSTATE_H */
