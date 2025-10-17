/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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

#include "AppleSCSIPDT03Emulator.h"

#include <IOKit/IOMemoryDescriptor.h>

#include <IOKit/scsi/SCSICommandOperationCodes.h>
#include <IOKit/scsi/SCSICmds_INQUIRY_Definitions.h>
#include <IOKit/scsi/SCSICmds_REPORT_LUNS_Definitions.h>
#include <IOKit/scsi/SCSICmds_READ_CAPACITY_Definitions.h>


//-----------------------------------------------------------------------------
//	Macros
//-----------------------------------------------------------------------------

#define DEBUG 												0
#define DEBUG_ASSERT_COMPONENT_NAME_STRING					"PDT03LUNEmulator"

#if DEBUG
#define EMULATOR_ADAPTER_DEBUGGING_LEVEL					4
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

#define super AppleSCSILogicalUnitEmulator
OSDefineMetaClassAndStructors ( AppleSCSIPDT03Emulator, AppleSCSILogicalUnitEmulator );


//-----------------------------------------------------------------------------
//	Globals
//-----------------------------------------------------------------------------

SCSICmd_INQUIRY_StandardData sInquiryData =
{
	kINQUIRY_PERIPHERAL_TYPE_ProcessorSPCDevice,	// PERIPHERAL_DEVICE_TYPE
	0,	// RMB;
	5,	// VERSION
	2,	// RESPONSE_DATA_FORMAT
	sizeof ( SCSICmd_INQUIRY_StandardData ) - 5,	// ADDITIONAL_LENGTH
	0,	// SCCSReserved
	0,	// flags1
	0,	// flags2
	"APPLE",
	"SCSI Emulator",
	"1.0",
};



//-----------------------------------------------------------------------------
//	Create
//-----------------------------------------------------------------------------

AppleSCSIPDT03Emulator *
AppleSCSIPDT03Emulator::Create ( void )
{
	
	AppleSCSIPDT03Emulator *	logicalUnit = NULL;
	bool						result		= false;
	
	STATUS_LOG ( ( "AppleSCSIPDT03Emulator::Create\n" ) );
	
	logicalUnit = OSTypeAlloc ( AppleSCSIPDT03Emulator );
	require_nonzero ( logicalUnit, ErrorExit );
	
	result = logicalUnit->init ( );
	require ( result, ReleaseLogicalUnit );
	
	return logicalUnit;
	
	
ReleaseLogicalUnit:
	
	
	logicalUnit->release ( );
	
	
ErrorExit:
	
	
	return NULL;
	
}


//-----------------------------------------------------------------------------
//	SendCommand
//-----------------------------------------------------------------------------

int
AppleSCSIPDT03Emulator::SendCommand (
	UInt8 *					cdb,
	UInt8 					cbdLen,
	IOMemoryDescriptor *	dataDesc,
	UInt64 *				dataLen,
	SCSITaskStatus *		scsiStatus,
	SCSI_Sense_Data *		senseBuffer,
	UInt8 *					senseBufferLen )
{
		
	STATUS_LOG ( ( "AppleSCSIPDT03Emulator::SendCommand, LUN = %qd\n", GetLogicalUnitNumber ( ) ) );
	
	switch ( cdb[0] )
	{
		
		case kSCSICmd_TEST_UNIT_READY:
		{	
			
			COMMAND_LOG ( ( "SCSI Command: TEST_UNIT_READY\n" ) );
			
			*scsiStatus = kSCSITaskStatus_GOOD;
			*dataLen = 0;
			break;
			
		}
		
		case kSCSICmd_INQUIRY:
		{
			
			COMMAND_LOG ( ( "SCSI Command: INQUIRY\n" ) );
			
			if ( cdb[1] == 1 )
			{
								
				COMMAND_LOG ( ( "INQUIRY VPD requested, PDT03 doesn't support it\n" ) );
				
				*scsiStatus = kSCSITaskStatus_CHECK_CONDITION;
				
				if ( senseBuffer != NULL )
				{
					
					UInt8	amount = min ( *senseBufferLen, sizeof ( SCSI_Sense_Data ) );
					
					bzero ( senseBuffer, *senseBufferLen );
					bcopy ( &gInvalidCDBFieldSenseData, senseBuffer, amount );
					
					*senseBufferLen = amount;
					
				}
				
			}
			
			else if ( ( cdb[1] == 2 ) || ( cdb[2] != 0 ) || ( cdb[3] != 0 ) )
			{
				
				COMMAND_LOG ( ( "Illegal request\n" ) );
				
				// Don't support CMDDT bit, or PAGE_CODE without EVPD set.
				*scsiStatus = kSCSITaskStatus_CHECK_CONDITION;

				if ( senseBuffer != NULL )
				{
					
					UInt8	amount = min ( *senseBufferLen, sizeof ( SCSI_Sense_Data ) );
					
					bzero ( senseBuffer, *senseBufferLen );
					bcopy ( &gInvalidCDBFieldSenseData, senseBuffer, amount );
					
					*senseBufferLen = amount;
					
				}
			
			}
			
			else
			{
				
				COMMAND_LOG ( ( "Standard INQUIRY\n" ) );
				
				*dataLen = min ( sizeof ( sInquiryData ), *dataLen );
				dataDesc->writeBytes ( 0, &sInquiryData, *dataLen );
				
				*scsiStatus = kSCSITaskStatus_GOOD;
				
			}
			
		}
		break;

		case kSCSICmd_REQUEST_SENSE:
		{
			
			COMMAND_LOG ( ( "SCSI Command: REQUEST_SENSE (desc = %s, allocation length = %d bytes) - returning CHECK CONDITION with INVALID COMMAND\n", (cdb[1] & 0x01) ? "TRUE" : "FALSE", cdb[4] ) );
			
			*scsiStatus = kSCSITaskStatus_CHECK_CONDITION;
			*dataLen = 0;
			
			if ( senseBuffer != NULL )
			{
				
				UInt8	amount = min ( *senseBufferLen, sizeof ( SCSI_Sense_Data ) );
				
				bzero ( senseBuffer, *senseBufferLen );
				bcopy ( &gInvalidCommandSenseData, senseBuffer, amount );
				
				*senseBufferLen = amount;
				
			}
			
		}
		break;


		default:
		{
			
			COMMAND_LOG ( ( "SCSI Command: Unknown: 0x%X\n", cdb[0] ) );
			
			*scsiStatus = kSCSITaskStatus_CHECK_CONDITION;
			
			if ( senseBuffer != NULL )
			{
				
				UInt8	amount = min ( *senseBufferLen, sizeof ( SCSI_Sense_Data ) );
				
				bzero ( senseBuffer, *senseBufferLen );
				bcopy ( &gInvalidCommandSenseData, senseBuffer, amount );
				
				*senseBufferLen = amount;
				
			}
			
		}
		break;
		
	}
	
	return 1;
	
}