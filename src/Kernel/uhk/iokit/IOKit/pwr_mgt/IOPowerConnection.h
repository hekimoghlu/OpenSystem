/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
 * Copyright (c) 1998 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 */


#ifndef _IOKIT_IOPOWERCONNECTION_H
#define _IOKIT_IOPOWERCONNECTION_H

#include <IOKit/IOService.h>
#include <IOKit/pwr_mgt/IOPM.h>

/*! @class IOPowerConnection
 *  Do not use IOPowerConnection. This class is an implementation detail defined
 *  for IOPM's management of the IORegistry IOPower plane.
 *
 *  Only Kernel IOKit power management should reference the IOPowerConnection class.
 */

class IOPowerConnection : public IOService
{
	OSDeclareDefaultStructors(IOPowerConnection);

protected:
/*! @field parentKnowsState	true: parent knows state of its domain
 *                                   used by child */
	bool            stateKnown;

/*! @field currentPowerFlags	power flags which describe  the current state of the power domain
 *                                   used by child */
	IOPMPowerFlags      currentPowerFlags;

/*! @field desiredDomainState	state number which corresponds to the child's desire
 *                                   used by parent */
	unsigned long       desiredDomainState;

/*! @field requestFlag		set to true when desiredDomainState is set */
	bool            requestFlag;

/*! @field preventIdleSleepFlag	true if child has this bit set in its desired state
 *                                   used by parent */
	unsigned long       preventIdleSleepFlag;

/*! @field preventSystemSleepFlag	true if child has this bit set in its desired state
 *                                   used by parent */
	unsigned long       preventSystemSleepFlag;

/*! @field awaitingAck		true if child has not yet acked our notification
 *                                   used by parent */
	bool            awaitingAck;

/*! @field readyFlag		true if the child has been added as a power child
 *                                   used by parent */
	bool            readyFlag;

#ifdef XNU_KERNEL_PRIVATE
public:
	bool            delayChildNotification;
#endif

public:
/*! @function setParentKnowsState
 *   @abstract Sets the stateKnown variable.
 *   @discussion Called by the parent when the object is created and called by the child when it discovers that the parent now knows its state. */
	void setParentKnowsState(bool );

/*! @function setParentCurrentPowerFlags
 *   @abstract Sets the currentPowerFlags variable.
 *   @discussion Called by the parent when the object is created and called by the child when it discovers that the parent state is changing. */
	void setParentCurrentPowerFlags(IOPMPowerFlags );

/*! @function parentKnowsState
 *   @abstract Returns the stateKnown variable. */
	bool parentKnowsState(void );

/*! @function parentCurrentPowerFlags
 *   @abstract Returns the currentPowerFlags variable. */
	IOPMPowerFlags parentCurrentPowerFlags(void );

/*! @function setDesiredDomainState
 *   @abstract Sets the desiredDomainState variable.
 *   @discussion Called by the parent. */
	void setDesiredDomainState(unsigned long );

/*! @function getDesiredDomainState
 *   @abstract Returns the desiredDomainState variable.
 *  @discussion Called by the parent. */
	unsigned long getDesiredDomainState( void );

/*! @function setChildHasRequestedPower
*   @abstract Set the flag that says that the child has called requestPowerDomainState.
*  @discussion Called by the parent. */
	void setChildHasRequestedPower( void );

/*! @function childHasRequestedPower
 *   @abstract Return the flag that says whether the child has called requestPowerDomainState.
 *  @discussion Called by the PCI Aux Power Supply Driver to see if a device driver
 *   is power managed. */
	bool childHasRequestedPower( void );

/*! @function setPreventIdleSleepFlag
 *   @abstract Sets the preventIdleSleepFlag variable.
 *   @discussion Called by the parent. */
	void setPreventIdleSleepFlag(unsigned long );

/*! @function getPreventIdleSleepFlag
 *   @abstract Returns the preventIdleSleepFlag variable.
 *  @discussion Called by the parent. */
	bool getPreventIdleSleepFlag( void );

/*! @function setPreventSystemSleepFlag
 *   @abstract Sets the preventSystemSleepFlag variable.
 *   @discussion Called by the parent. */
	void setPreventSystemSleepFlag(unsigned long );

/*! @function getPreventSystemSleepFlag
 *   @abstract Returns the preventSystemSleepFlag variable.
 *   @discussion Called by the parent. */
	bool getPreventSystemSleepFlag( void );

/*! @function setAwaitingAck
 *   @abstract Sets the awaitingAck variable.
 *   @discussion Called by the parent. */
	void setAwaitingAck( bool );

/*! @function getAwaitingAck
 *   @abstract Returns the awaitingAck variable.
 *   @discussion Called by the parent. */
	bool getAwaitingAck( void );

/*! @function setReadyFlag
 *   @abstract Sets the readyFlag variable.
 *   @discussion Called by the parent. */
	void setReadyFlag( bool flag );

/*! @function getReadyFlag
 *   @abstract Returns the readyFlag variable.
 *   @discussion Called by the parent. */
	bool getReadyFlag( void ) const;
};

#endif /* ! _IOKIT_IOPOWERCONNECTION_H */
