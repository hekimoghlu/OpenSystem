/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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
#ifndef __APPLE_SCSI_EMULATOR_ADAPTER_UC_H__
#define __APPLE_SCSI_EMULATOR_ADAPTER_UC_H__


//-----------------------------------------------------------------------------
//	Includes
//-----------------------------------------------------------------------------

// SCSI Architecture Model Family includes
#include <IOKit/scsi/SCSITask.h>
#include <mach/vm_types.h>


//-----------------------------------------------------------------------------
//	Constants
//-----------------------------------------------------------------------------

#define kSCSIEmulatorAdapterUserClientAccessMask	0x1000
#define kSCSIEmulatorAdapterUserClientConnection	15
#define kInitiatorID								15

enum
{
	kUserClientCreateLUN		= 0,
	kUserClientDestroyLUN		= 1,
	kUserClientDestroyTarget	= 2,
	kUserClientMethodCount
};


//-----------------------------------------------------------------------------
//	Structures
//-----------------------------------------------------------------------------

#pragma pack(1)
typedef struct EmulatorLUNParamsStruct
{
	SCSILogicalUnitNumber	logicalUnit;
	UInt64					capacity;
	mach_vm_address_t		inquiryData;
	UInt32					inquiryDataLength;
	mach_vm_address_t		inquiryPage00Data;
	UInt32					inquiryPage00DataLength;
	mach_vm_address_t		inquiryPage80Data;
	UInt32					inquiryPage80DataLength;
	mach_vm_address_t		inquiryPage83Data;
	UInt32					inquiryPage83DataLength;
} EmulatorLUNParamsStruct;

typedef struct EmulatorTargetParamsStruct
{
	SCSITargetIdentifier	targetID;
	EmulatorLUNParamsStruct	lun;
} EmulatorTargetParamsStruct;

#pragma options align=reset


#if defined(KERNEL) && defined(__cplusplus)

//-----------------------------------------------------------------------------
//	Includes
//-----------------------------------------------------------------------------

#include <IOKit/IOCommandGate.h>
#include <IOKit/IOService.h>
#include <IOKit/IOUserClient.h>
#include <IOKit/IOWorkLoop.h>


//-----------------------------------------------------------------------------
//	Class Declaration
//-----------------------------------------------------------------------------

class AppleSCSIEmulatorAdapterUserClient : public IOUserClient
{
	
	OSDeclareDefaultStructors ( AppleSCSIEmulatorAdapterUserClient )
	
protected:
	
	bool 	initWithTask 		( task_t 			owningTask,
								  void *			securityToken,
								  UInt32			type,
								  OSDictionary *	properties );
	
    bool 		start 			( IOService * provider );
	
	void		free			( void );
	
	IOReturn	clientClose		( void );
	
	bool		finalize		( IOOptionBits options );
	
	IOReturn	externalMethod (
					uint32_t						selector,
					IOExternalMethodArguments * 	args,
					IOExternalMethodDispatch * 		dispatch,
					OSObject *						target,
					void *							reference );
	
private:
	
	task_t					fTask;
	IOService *				fProvider;
	IOCommandGate *			fCommandGate;
	IOWorkLoop *			fWorkLoop;
	
};


#endif	/* #if defined(KERNEL) && defined(__cplusplus) */

#endif	/* __APPLE_SCSI_EMULATOR_ADAPTER_UC_H__ */