/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
#ifndef __APPLE_SCSI_TARGET_EMULATOR_H__
#define __APPLE_SCSI_TARGET_EMULATOR_H__


//-----------------------------------------------------------------------------
//	Includes
//-----------------------------------------------------------------------------

#include <IOKit/IOMemoryDescriptor.h>
#include <IOKit/scsi/SCSITask.h>
#include <libkern/c++/OSArray.h>
#include <IOKit/IOLocks.h>
#include <IOKit/scsi/SCSICmds_REQUEST_SENSE_Defs.h>
#include <IOKit/scsi/SCSICmds_REPORT_LUNS_Definitions.h>
#include "AppleSCSIEmulatorDefines.h"


//-----------------------------------------------------------------------------
//	Constants
//-----------------------------------------------------------------------------

enum
{
	kTargetStateChangeActiveBit			= 0,
	kTargetStateChangeActiveWaitBit		= 1,
	
	kTargetStateChangeActiveMask		= (1 << kTargetStateChangeActiveBit),
	kTargetStateChangeActiveWaitMask	= (1 << kTargetStateChangeActiveWaitBit),
};


//-----------------------------------------------------------------------------
//	Class declaration
//-----------------------------------------------------------------------------

class AppleSCSITargetEmulator : public OSObject
{

	OSDeclareDefaultStructors ( AppleSCSITargetEmulator )

public:
	
	static AppleSCSITargetEmulator *	Create ( SCSITargetIdentifier targetID );
	
	inline SCSITargetIdentifier GetTargetID ( void ) { return fTargetID; }
	
	bool	Init ( SCSITargetIdentifier targetID );
	
	bool	AddLogicalUnit (
		SCSILogicalUnitNumber 	logicalUnitNumber,
		UInt64					capacity,
		IOMemoryDescriptor * 	inquiryBuffer,
		IOMemoryDescriptor * 	inquiryPage00Buffer,
		IOMemoryDescriptor * 	inquiryPage80Buffer,
		IOMemoryDescriptor * 	inquiryPage83Buffer );

	void	RemoveLogicalUnit ( SCSILogicalUnitNumber logicalUnitNumber );
	void	RebuildListOfLUNs ( void );
	
	void	free ( void );
	
#if USE_LUN_BYTES
	
	int SendCommand ( UInt8 *				cdb,
					  UInt8					cbdLen,
					  IOMemoryDescriptor * 	dataDesc,
					  UInt64 *				dataLen,
					  SCSILogicalUnitBytes	logicalUnitBytes,
					  SCSITaskStatus * 		scsiStatus,
					  SCSI_Sense_Data * 	senseBuffer,
					  UInt8 *				senseBufferLen );
	
#else
	
	int SendCommand ( UInt8 *				cdb,
					  UInt8					cbdLen,
					  IOMemoryDescriptor * 	dataDesc,
					  UInt64 *				dataLen,
					  SCSILogicalUnitNumber	logicalUnitNumber,
					  SCSITaskStatus * 		scsiStatus,
					  SCSI_Sense_Data * 	senseBuffer,
					  UInt8 *				senseBufferLen );

#endif	/* USE_LUN_BYTES */
	
	
	static SCSI_Sense_Data			sBadLUNSenseData;
	static SCSI_Sense_Data			sLUNInventoryChangedData;
	
	SCSITargetIdentifier 			fTargetID;
	SCSICmd_REPORT_LUNS_Header *	fLUNReportBuffer;
	UInt32							fLUNReportBufferSize;
	UInt32							fLUNDataAvailable;
	OSOrderedSet *					fLUNs;
	IOLock *						fLock;
	UInt32							fState;
	bool							fLUNInventoryChanged;
	
};


#endif	/* __APPLE_SCSI_TARGET_EMULATOR_H__ */