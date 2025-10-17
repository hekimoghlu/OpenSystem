/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
#ifndef __APPLE_SCSI_LOGICAL_UNIT_0_EMULATOR_H__
#define __APPLE_SCSI_LOGICAL_UNIT_0_EMULATOR_H__


//-----------------------------------------------------------------------------
//	Includes
//-----------------------------------------------------------------------------

#include <IOKit/IOMemoryDescriptor.h>
#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/scsi/SCSITask.h>
#include <IOKit/scsi/SCSICmds_REQUEST_SENSE_Defs.h>
#include <IOKit/scsi/SCSICmds_INQUIRY_Definitions.h>

#include "AppleSCSILogicalUnitEmulator.h"


//-----------------------------------------------------------------------------
//	Class Declaration
//-----------------------------------------------------------------------------

class AppleSCSIPDT03Emulator : public AppleSCSILogicalUnitEmulator
{
	
	OSDeclareDefaultStructors ( AppleSCSIPDT03Emulator )
	
public:
	
	static AppleSCSIPDT03Emulator * Create ( void );
	
	inline SCSILogicalUnitNumber GetLogicalUnitNumber ( void ) { return 0; }
	
	int SendCommand ( UInt8 *				cdb,
					  UInt8					cbdLen,
					  IOMemoryDescriptor * 	dataDesc,
					  UInt64 *				dataLen,
					  SCSITaskStatus * 		scsiStatus,
					  SCSI_Sense_Data * 	senseBuffer,
					  UInt8 *				senseBufferLen );
		
	static SCSI_Sense_Data		sInvalidCommandSenseData;
	static SCSI_Sense_Data		sInvalidCDBFieldSenseData;
	
};


#endif	/* __APPLE_SCSI_LOGICAL_UNIT_0_EMULATOR_H__ */