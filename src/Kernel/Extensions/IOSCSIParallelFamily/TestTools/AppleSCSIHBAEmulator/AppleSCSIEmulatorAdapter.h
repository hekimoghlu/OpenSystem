/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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
// Code to implement a basic Parallel Tasking SCSI HBA.  In this case, a SCSI-based RAM disk.

#ifndef __APPLE_SCSI_EMULATOR_ADAPTER_H__
#define __APPLE_SCSI_EMULATOR_ADAPTER_H__


//-----------------------------------------------------------------------------
//	Includes
//-----------------------------------------------------------------------------

#include <IOKit/scsi/spi/IOSCSIParallelInterfaceController.h>
#include <IOKit/scsi/SCSITask.h>

#include "AppleSCSIEmulatorAdapterUC.h"

// Forward declarations
class AppleSCSIEmulatorEventSource;


//-----------------------------------------------------------------------------
//	Class declaration
//-----------------------------------------------------------------------------

class AppleSCSIEmulatorAdapter : public IOSCSIParallelInterfaceController
{

	OSDeclareDefaultStructors ( AppleSCSIEmulatorAdapter )
	
public:
    
	SCSILogicalUnitNumber	ReportHBAHighestLogicalUnitNumber ( void );
	
	bool	DoesHBASupportSCSIParallelFeature ( 
							SCSIParallelFeature 		theFeature );
	
	bool	InitializeTargetForID (  
							SCSITargetIdentifier 		targetID );
	
	SCSIServiceResponse	AbortTaskRequest ( 	
							SCSITargetIdentifier 		theT,
							SCSILogicalUnitNumber		theL,
							SCSITaggedTaskIdentifier	theQ );
	
	SCSIServiceResponse AbortTaskSetRequest (
							SCSITargetIdentifier 		theT,
							SCSILogicalUnitNumber		theL );
	
	SCSIServiceResponse ClearACARequest (
							SCSITargetIdentifier 		theT,
							SCSILogicalUnitNumber		theL );
	
	SCSIServiceResponse ClearTaskSetRequest (
							SCSITargetIdentifier 		theT,
							SCSILogicalUnitNumber		theL );
	
	SCSIServiceResponse LogicalUnitResetRequest (
							SCSITargetIdentifier 		theT,
							SCSILogicalUnitNumber		theL );
	
	SCSIServiceResponse TargetResetRequest (
							SCSITargetIdentifier 		theT );
	
	// Methods the user client calls
	IOReturn	CreateLUN ( EmulatorTargetParamsStruct * targetParameters, task_t task );
	IOReturn	DestroyLUN ( SCSITargetIdentifier targetID, SCSILogicalUnitNumber logicalUnit );
	IOReturn	DestroyTarget ( SCSITargetIdentifier targetID );
	
	
protected:
	
	void SetControllerProperties ( void );
	
	void TaskComplete ( SCSIParallelTaskIdentifier parallelRequest );

	void CompleteTaskOnWorkloopThread (
							SCSIParallelTaskIdentifier		parallelRequest,
							bool							transportSuccessful,
							SCSITaskStatus					scsiStatus,
							UInt64							actuallyTransferred,
							SCSI_Sense_Data *				senseBuffer,
							UInt8							senseLength );
	
	SCSIInitiatorIdentifier	ReportInitiatorIdentifier ( void );
	
	SCSIDeviceIdentifier	ReportHighestSupportedDeviceID ( void );
		
	UInt32		ReportMaximumTaskCount ( void );
		
	UInt32		ReportHBASpecificTaskDataSize ( void );
	
	UInt32		ReportHBASpecificDeviceDataSize ( void );
	
	void		ReportHBAConstraints ( OSDictionary * constraints );
	
	bool		DoesHBAPerformDeviceManagement ( void );

	bool	InitializeController ( void );
	
	void	TerminateController ( void );
	
	bool	StartController ( void );
		
	void	StopController ( void );
	
	void	HandleInterruptRequest ( void );

	SCSIServiceResponse ProcessParallelTask (
							SCSIParallelTaskIdentifier parallelRequest );
	
	IOInterruptEventSource * CreateDeviceInterrupt ( 
											IOInterruptEventSource::Action			action,
											IOFilterInterruptEventSource::Filter	filter,
											IOService *								provider );
	
private:
	
	AppleSCSIEmulatorEventSource *	fEventSource;
	OSArray *						fTargetEmulators;
	
};


#endif	/* __APPLE_SCSI_EMULATOR_ADAPTER_H__ */