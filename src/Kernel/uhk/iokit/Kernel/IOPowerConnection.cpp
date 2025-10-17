/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
#include <IOKit/pwr_mgt/IOPowerConnection.h>

#define super IOService
OSDefineMetaClassAndStructors(IOPowerConnection, IOService)


// **********************************************************************************
// setDesiredDomainState
//
// Parent of the connection calls here to save the childs desire
// **********************************************************************************
void
IOPowerConnection::setDesiredDomainState(unsigned long stateNumber )
{
	desiredDomainState = stateNumber;
}


// **********************************************************************************
// getDesiredDomainState
//
// **********************************************************************************
unsigned long
IOPowerConnection::getDesiredDomainState( void )
{
	return desiredDomainState;
}


// **********************************************************************************
// setChildHasRequestedPower
//
// Parent of the connection calls here when the child requests power
// **********************************************************************************
void
IOPowerConnection::setChildHasRequestedPower( void )
{
	requestFlag = true;
}

// **********************************************************************************
// childHasRequestedPower
//
// Parent of the connection calls here when the child requests power
// **********************************************************************************
bool
IOPowerConnection::childHasRequestedPower( void )
{
	return requestFlag;
}


// **********************************************************************************
// setPreventIdleSleepFlag
//
// **********************************************************************************
void
IOPowerConnection::setPreventIdleSleepFlag( unsigned long flag )
{
	preventIdleSleepFlag = (flag != 0);
}


// **********************************************************************************
// getPreventIdleSleepFlag
//
// **********************************************************************************
bool
IOPowerConnection::getPreventIdleSleepFlag( void )
{
	return preventIdleSleepFlag;
}


// **********************************************************************************
// setPreventSystemSleepFlag
//
// **********************************************************************************
void
IOPowerConnection::setPreventSystemSleepFlag( unsigned long flag )
{
	preventSystemSleepFlag = (flag != 0);
}


// **********************************************************************************
// getPreventSystemSleepFlag
//
// **********************************************************************************
bool
IOPowerConnection::getPreventSystemSleepFlag( void )
{
	return preventSystemSleepFlag;
}


// **********************************************************************************
// setParentKnowsState
//
// Child of the connection calls here to set its reminder that the parent does
// or does not yet know the state if its domain.
// **********************************************************************************
void
IOPowerConnection::setParentKnowsState(bool flag )
{
	stateKnown = flag;
}


// **********************************************************************************
// setParentCurrentPowerFlags
//
// Child of the connection calls here to save what the parent says
// is the state if its domain.
// **********************************************************************************
void
IOPowerConnection::setParentCurrentPowerFlags(IOPMPowerFlags flags )
{
	currentPowerFlags = flags;
}


// **********************************************************************************
// parentKnowsState
//
// **********************************************************************************
bool
IOPowerConnection::parentKnowsState(void )
{
	return stateKnown;
}


// **********************************************************************************
// parentCurrentPowerFlags
//
// **********************************************************************************
IOPMPowerFlags
IOPowerConnection::parentCurrentPowerFlags(void )
{
	return currentPowerFlags;
}


// **********************************************************************************
// setAwaitingAck
//
// **********************************************************************************
void
IOPowerConnection::setAwaitingAck( bool value )
{
	awaitingAck = value;
}


// **********************************************************************************
// getAwaitingAck
//
// **********************************************************************************
bool
IOPowerConnection::getAwaitingAck( void )
{
	return awaitingAck;
}


// **********************************************************************************
// setReadyFlag
//
// **********************************************************************************
void
IOPowerConnection::setReadyFlag( bool flag )
{
	readyFlag = flag;
}


// **********************************************************************************
// getReadyFlag
//
// **********************************************************************************
bool
IOPowerConnection::getReadyFlag( void ) const
{
	return readyFlag;
}
