/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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
#ifndef _IOKIT_IOSERVICEPM_H
#define _IOKIT_IOSERVICEPM_H

#include <IOKit/pwr_mgt/IOPM.h>

class IOService;
class IOServicePM;
class IOPowerConnection;
class IOWorkLoop;
class IOCommandGate;
class IOTimerEventSource;
class IOPlatformExpert;

#ifdef XNU_KERNEL_PRIVATE
class IOPMinformee;
class IOPMinformeeList;
class IOPMWorkQueue;
class IOPMRequest;
class IOPMRequestQueue;
class IOPMCompletionQueue;

typedef void (*IOPMCompletionAction)(void * target, void * param);

// PM channels for IOReporting
#ifndef kPMPowerStatesChID
#define kPMPowerStatesChID  IOREPORT_MAKEID('P','M','S','t','H','i','s','t')
#endif

#ifndef kPMCurrStateChID
#define kPMCurrStateChID  IOREPORT_MAKEID( 'P','M','C','u','r','S','t','\0' )
#endif

// state_id details in PM channels
#define kPMReportPowerOn       0x01
#define kPMReportDeviceUsable  0x02
#define kPMReportLowPower      0x04


typedef unsigned long       IOPMPowerStateIndex;
typedef uint32_t            IOPMPowerChangeFlags;
typedef uint32_t            IOPMRequestTag;

struct IOPMDriverCallEntry {
	queue_chain_t   link;
	thread_t        thread;
	IOService *     target;
	const void  *callMethod;
};

// Power clients (desires)
extern const OSSymbol *     gIOPMPowerClientDevice;
extern const OSSymbol *     gIOPMPowerClientDriver;
extern const OSSymbol *     gIOPMPowerClientChildProxy;
extern const OSSymbol *     gIOPMPowerClientChildren;
extern const OSSymbol *     gIOPMPowerClientRootDomain;

/* Binary compatibility with drivers that access pm_vars */
#ifdef __LP64__
#define PM_VARS_SUPPORT     0
#else
#define PM_VARS_SUPPORT     1
#endif

#if PM_VARS_SUPPORT
/* Deprecated in version 10.5 */
class IOPMprot : public OSObject
{
	friend class IOService;

	OSDeclareDefaultStructors(IOPMprot);

public:
	const char *            ourName;
	IOPlatformExpert *      thePlatform;
	unsigned long           theNumberOfPowerStates;
	IOPMPowerState          thePowerStates[IOPMMaxPowerStates];
	IOService *             theControllingDriver;
	unsigned long           aggressiveness;
	unsigned long           current_aggressiveness_values[kMaxType + 1];
	bool                    current_aggressiveness_valid[kMaxType + 1];
	unsigned long           myCurrentState;
};
#endif /* PM_VARS_SUPPORT */
#endif /* XNU_KERNEL_PRIVATE */
#endif /* !_IOKIT_IOSERVICEPM_H */
