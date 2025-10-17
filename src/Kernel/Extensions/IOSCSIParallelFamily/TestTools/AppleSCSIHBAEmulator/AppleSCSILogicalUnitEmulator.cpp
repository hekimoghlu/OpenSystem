/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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
//-----------------------------------------------------------------------------
//	Includes
//-----------------------------------------------------------------------------

#include "AppleSCSILogicalUnitEmulator.h"

#include <IOKit/IOMemoryDescriptor.h>

#include <IOKit/scsi/SCSICommandOperationCodes.h>
#include <IOKit/scsi/SCSICmds_INQUIRY_Definitions.h>
#include <IOKit/scsi/SCSICmds_REPORT_LUNS_Definitions.h>
#include <IOKit/scsi/SCSICmds_READ_CAPACITY_Definitions.h>


//-----------------------------------------------------------------------------
//	Macros
//-----------------------------------------------------------------------------

#define DEBUG 												1
#define DEBUG_ASSERT_COMPONENT_NAME_STRING					"LUNEmulator"

#if DEBUG
#define EMULATOR_ADAPTER_DEBUGGING_LEVEL					2
#endif

#include "DebugSupport.h"

#if ( EMULATOR_ADAPTER_DEBUGGING_LEVEL >= 1 )
#define PANIC_NOW(x)		panic x
#else
#define PANIC_NOW(x)		
#endif

#if ( EMULATOR_ADAPTER_DEBUGGING_LEVEL >= 2 )
#define ERROR_LOG(x)		IOLog x; IOSleep(1)
#else
#define ERROR_LOG(x)		
#endif

#if ( EMULATOR_ADAPTER_DEBUGGING_LEVEL >= 3 )
#define STATUS_LOG(x)		IOLog x; IOSleep(1)
#else
#define STATUS_LOG(x)
#endif

#if ( EMULATOR_ADAPTER_DEBUGGING_LEVEL >= 4 )
#define COMMAND_LOG(x)		IOLog x; IOSleep(1)
#else
#define COMMAND_LOG(x)
#endif


#define super OSObject
OSDefineMetaClass ( AppleSCSILogicalUnitEmulator, OSObject );
OSDefineAbstractStructors ( AppleSCSILogicalUnitEmulator, OSObject );


//-----------------------------------------------------------------------------
//	Globals
//-----------------------------------------------------------------------------

SCSI_Sense_Data gInvalidCDBFieldSenseData =
{
	/* VALID_RESPONSE_CODE */				kSENSE_DATA_VALID | kSENSE_RESPONSE_CODE_Current_Errors,
	/* SEGMENT_NUMBER */					0x00, // Obsolete
	/* SENSE_KEY */							kSENSE_KEY_ILLEGAL_REQUEST,
	/* INFORMATION_1 */						0x00,
	/* INFORMATION_2 */						0x00,
	/* INFORMATION_3 */						0x00,
	/* INFORMATION_4 */						0x00,
	/* ADDITIONAL_SENSE_LENGTH */			0x00,
	/* COMMAND_SPECIFIC_INFORMATION_1 */	0x00,
	/* COMMAND_SPECIFIC_INFORMATION_2 */	0x00,
	/* COMMAND_SPECIFIC_INFORMATION_3 */	0x00,
	/* COMMAND_SPECIFIC_INFORMATION_4 */	0x00,
	/* ADDITIONAL_SENSE_CODE */				0x24, // INVALID FIELD IN CDB
	/* ADDITIONAL_SENSE_CODE_QUALIFIER */	0x00,
	/* FIELD_REPLACEABLE_UNIT_CODE */		0x00,
	/* SKSV_SENSE_KEY_SPECIFIC_MSB */		0x00,
	/* SENSE_KEY_SPECIFIC_MID */			0x00,
	/* SENSE_KEY_SPECIFIC_LSB */			0x00
};

SCSI_Sense_Data gInvalidCommandSenseData =
{
	/* VALID_RESPONSE_CODE */				kSENSE_DATA_VALID | kSENSE_RESPONSE_CODE_Current_Errors,
	/* SEGMENT_NUMBER */					0x00, // Obsolete
	/* SENSE_KEY */							kSENSE_KEY_ILLEGAL_REQUEST,
	/* INFORMATION_1 */						0x00,
	/* INFORMATION_2 */						0x00,
	/* INFORMATION_3 */						0x00,
	/* INFORMATION_4 */						0x00,
	/* ADDITIONAL_SENSE_LENGTH */			0x00,
	/* COMMAND_SPECIFIC_INFORMATION_1 */	0x00,
	/* COMMAND_SPECIFIC_INFORMATION_2 */	0x00,
	/* COMMAND_SPECIFIC_INFORMATION_3 */	0x00,
	/* COMMAND_SPECIFIC_INFORMATION_4 */	0x00,
	/* ADDITIONAL_SENSE_CODE */				0x20, // Invalid command code
	/* ADDITIONAL_SENSE_CODE_QUALIFIER */	0x00,
	/* FIELD_REPLACEABLE_UNIT_CODE */		0x00,
	/* SKSV_SENSE_KEY_SPECIFIC_MSB */		0x00,
	/* SENSE_KEY_SPECIFIC_MID */			0x00,
	/* SENSE_KEY_SPECIFIC_LSB */			0x00
};


//-----------------------------------------------------------------------------
//	SetLogicalUnitNumber
//-----------------------------------------------------------------------------

void
AppleSCSILogicalUnitEmulator::SetLogicalUnitNumber ( SCSILogicalUnitNumber logicalUnitNumber )
{

#if USE_LUN_BYTES	
	bzero ( fLogicalUnitBytes, sizeof ( SCSILogicalUnitBytes ) );
	
	if ( logicalUnitNumber < 256 )
	{
		
		fLogicalUnitBytes[0] = 0;
		fLogicalUnitBytes[1] = logicalUnitNumber & 0xFF;
		
	}
	
	else if ( logicalUnitNumber < 16384 )
	{
		
		fLogicalUnitBytes[0] = ( kREPORT_LUNS_ADDRESS_METHOD_FLAT_SPACE << 6 ) | ( ( logicalUnitNumber >> 8 ) & 0xFF );
		fLogicalUnitBytes[1] = logicalUnitNumber & 0xFF;
		
	}

#endif
	
	fLogicalUnitNumber = logicalUnitNumber;
	
}


//-----------------------------------------------------------------------------
//	GetLogicalUnitNumber
//-----------------------------------------------------------------------------

SCSILogicalUnitNumber
AppleSCSILogicalUnitEmulator::GetLogicalUnitNumber ( void )
{
	return fLogicalUnitNumber;
}


#if USE_LUN_BYTES

//-----------------------------------------------------------------------------
//	GetLogicalUnitBytes
//-----------------------------------------------------------------------------

void
AppleSCSILogicalUnitEmulator::GetLogicalUnitBytes ( SCSILogicalUnitBytes * logicalUnitBytes )
{
	bcopy ( fLogicalUnitBytes, logicalUnitBytes, sizeof ( SCSILogicalUnitBytes ) );
}

#endif	/* USE_LUN_BYTES */