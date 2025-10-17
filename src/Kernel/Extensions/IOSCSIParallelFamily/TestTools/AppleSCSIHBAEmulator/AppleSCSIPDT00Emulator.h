/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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
#ifndef __APPLE_SCSI_PDT00_EMULATOR_H__
#define __APPLE_SCSI_PDT00_EMULATOR_H__


//-----------------------------------------------------------------------------
//	Includes
//-----------------------------------------------------------------------------

#include <IOKit/IOMemoryDescriptor.h>
#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/scsi/SCSITask.h>
#include <IOKit/scsi/SCSICmds_REQUEST_SENSE_Defs.h>
#include <IOKit/scsi/SCSICmds_MODE_Definitions.h>
#include <IOKit/scsi/SCSICmds_INQUIRY_Definitions.h>

#include "AppleSCSILogicalUnitEmulator.h"


//-----------------------------------------------------------------------------
//	Constants
//-----------------------------------------------------------------------------

#define kTwentyMegabytes	(20 * 1024 * 1024)
#define kBlockSize			512

extern SCSICmd_INQUIRY_StandardData		gInquiryData;


//-----------------------------------------------------------------------------
//	Class declaration
//-----------------------------------------------------------------------------

class AppleSCSIPDT00Emulator : public AppleSCSILogicalUnitEmulator
{
	
	OSDeclareDefaultStructors ( AppleSCSIPDT00Emulator )
	
public:
	
	static AppleSCSIPDT00Emulator *
	WithCapacity ( UInt64 capacity = kTwentyMegabytes );

	bool	InitWithCapacity ( UInt64 capacity );
	
	bool SetDeviceBuffers ( 
		IOMemoryDescriptor * 	inquiryBuffer,
		IOMemoryDescriptor * 	inquiryPage00Buffer,
		IOMemoryDescriptor * 	inquiryPage80Buffer,
		IOMemoryDescriptor * 	inquiryPage83Buffer );
	
	void free ( void );
	
	int SendCommand ( UInt8 *				cdb,
					  UInt8					cbdLen,
					  IOMemoryDescriptor * 	dataDesc,
					  UInt64 *				dataLen,
					  SCSITaskStatus * 		scsiStatus,
					  SCSI_Sense_Data * 	senseBuffer,
					  UInt8 *				senseBufferLen );
	
private:
	
	UInt64						fBufferSize;
	IOBufferMemoryDescriptor *	fMemoryBuffer;
	UInt8 *						fMemory;
	
	UInt8 *						fInquiryData;
	UInt32						fInquiryDataSize;
	
	UInt8 *						fInquiryPage00Data;
	UInt32						fInquiryPage00DataSize;
	
	UInt8 *						fInquiryPage80Data;
	UInt32						fInquiryPage80DataSize;
	
	UInt8 *						fInquiryPage83Data;
	UInt32						fInquiryPage83DataSize;
	
};


#endif	/* __APPLE_SCSI_PDT00_EMULATOR_H__ */