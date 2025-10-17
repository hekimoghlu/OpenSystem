/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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

#include "Probing.h"

#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/storage/IOStorageProtocolCharacteristics.h>
#include <IOKit/scsi/SCSITask.h>


//-----------------------------------------------------------------------------
//	Macros
//-----------------------------------------------------------------------------

#define DEBUG 0

#define DEBUG_ASSERT_COMPONENT_NAME_STRING "Probing"

#if DEBUG
#define DEBUG_ASSERT_MESSAGE(componentNameString,	\
							 assertionString,		\
							 exceptionLabelString,	\
							 errorString,			\
							 fileName,				\
							 lineNumber,			\
							 errorCode)				\
DebugAssert(componentNameString,					\
					   assertionString,				\
					   exceptionLabelString,		\
					   errorString,					\
					   fileName,					\
					   lineNumber,					\
					   errorCode)					\

static void
DebugAssert ( const char *	componentNameString,
			  const char *	assertionString,
			  const char *	exceptionLabelString,
			  const char *	errorString,
			  const char *	fileName,
			  long			lineNumber,
			  int			errorCode )
{
	
	if ( ( assertionString != NULL ) && ( *assertionString != '\0' ) )
		printf ( "Assertion failed: %s: %s\n", componentNameString, assertionString );
	else
		printf ( "Check failed: %s:\n", componentNameString );
	if ( exceptionLabelString != NULL )
		printf ( "	 %s\n", exceptionLabelString );
	if ( errorString != NULL )
		printf ( "	 %s\n", errorString );
	if ( fileName != NULL )
		printf ( "	 file: %s\n", fileName );
	if ( lineNumber != 0 )
		printf ( "	 line: %ld\n", lineNumber );
	if ( errorCode != 0 )
		printf ( "	 error: %d\n", errorCode );
	
}

#endif	/* DEBUG */

#include <AssertMacros.h>


//-----------------------------------------------------------------------------
//	Constants
//-----------------------------------------------------------------------------

#define kIOSCSIParallelInterfaceControllerClassString	"IOSCSIParallelInterfaceController"


//-----------------------------------------------------------------------------
//	Prototypes
//-----------------------------------------------------------------------------

static IOReturn
ReprobeTargetDevice ( io_service_t controller, SCSITargetIdentifier targetID );


//-----------------------------------------------------------------------------
//	ReprobeDomainTarget - Reprobes device at targetID on a SCSI Domain
//-----------------------------------------------------------------------------


IOReturn
ReprobeDomainTarget ( UInt64				domainID,
					  SCSITargetIdentifier	targetID )
{
	
	IOReturn			result		= kIOReturnSuccess;
	io_service_t		service		= MACH_PORT_NULL;
	io_iterator_t		iterator	= MACH_PORT_NULL;
	boolean_t			found		= false;
	
	// First, let's find all the SCSI Parallel Controllers.
	result = IOServiceGetMatchingServices ( kIOMasterPortDefault,
											IOServiceMatching ( kIOSCSIParallelInterfaceControllerClassString ),
											&iterator );
	
	require ( ( result == kIOReturnSuccess ), ErrorExit );
	
	service = IOIteratorNext ( iterator );
	while ( service != MACH_PORT_NULL )
	{
		
		// Have we found the one with the specified domainID yet?
		if ( found == false )
		{
			
			CFMutableDictionaryRef	deviceDict	= NULL;
			CFDictionaryRef			subDict		= NULL;
			
			// Get the properties for this node from the IORegistry
			result = IORegistryEntryCreateCFProperties ( service,
														 &deviceDict,
														 kCFAllocatorDefault,
														 0 );
			
			// Get the protocol characteristics dictionary
			subDict = ( CFDictionaryRef ) CFDictionaryGetValue ( deviceDict, CFSTR ( kIOPropertyProtocolCharacteristicsKey ) );
			if ( subDict != NULL )
			{
				
				CFNumberRef		deviceDomainIDRef = 0;
				
				// Get the SCSI Domain Identifier value
				deviceDomainIDRef = ( CFNumberRef ) CFDictionaryGetValue ( subDict, CFSTR ( kIOPropertySCSIDomainIdentifierKey ) );
				if ( deviceDomainIDRef != 0 )
				{
					
					UInt64	deviceDomainID = 0;
					
					// Get the value from the CFNumberRef.
					if ( CFNumberGetValue ( deviceDomainIDRef, kCFNumberLongLongType, &deviceDomainID ) )
					{
						
						// Does the domainID match?
						if ( domainID == deviceDomainID )
						{
							
							// Find the target device and reprobe it.
							result = ReprobeTargetDevice ( service, targetID );
							found = true;
							
						}
						
					}
				
				}
				
			}
			
			if ( deviceDict != NULL )
				CFRelease ( deviceDict );
			
		}
		
		IOObjectRelease ( service );
		
		service = IOIteratorNext ( iterator );
		
	}
	
	IOObjectRelease ( iterator );
	iterator = MACH_PORT_NULL;
	
	if ( found == false )
		result = kIOReturnNoDevice;
	
	
ErrorExit:
	
	
	return result;
	
}


//-----------------------------------------------------------------------------
//	ReprobeTargetDevice - 	Actually performs the reprobe if it can find the
//							IOSCSIParallelInterfaceDevice at the targetID.
//-----------------------------------------------------------------------------

static IOReturn
ReprobeTargetDevice ( io_service_t controller, SCSITargetIdentifier targetID )
{
	
	IOReturn		result 		= kIOReturnSuccess;
	io_iterator_t	childIter	= MACH_PORT_NULL;
	io_service_t	service		= MACH_PORT_NULL;
	boolean_t		found		= false;
	
	// We find the children for the controller and iterate over them looking for the
	// one which has a targetID which matches.
	result = IORegistryEntryGetChildIterator ( controller, kIOServicePlane, &childIter );
	require ( ( result == kIOReturnSuccess ), ErrorExit );
	
	service = IOIteratorNext ( childIter );	
	while ( service != MACH_PORT_NULL )
	{

		// Did we find our device yet? If not, then try to find it. If we have already
		// found it, we still need to call IOObjectRelease on the io_service_t
		// or it will have an artificial retain count on it.
		if ( found == false )
		{
			
			CFMutableDictionaryRef	deviceDict	= NULL;
			CFDictionaryRef			subDict		= NULL;
			
			// Get the properties for this node from the IORegistry
			result = IORegistryEntryCreateCFProperties ( service,
														 &deviceDict,
														 kCFAllocatorDefault,
														 0 );
			
			// Get the protocol characteristics dictionary
			subDict = ( CFDictionaryRef ) CFDictionaryGetValue ( deviceDict, CFSTR ( kIOPropertyProtocolCharacteristicsKey ) );
			if ( subDict != NULL )
			{
				
				CFNumberRef		deviceTargetIDRef = 0;
				
				// Get the targetID value
				deviceTargetIDRef = ( CFNumberRef ) CFDictionaryGetValue ( subDict, CFSTR ( kIOPropertySCSITargetIdentifierKey ) );
				if ( deviceTargetIDRef != 0 )
				{
					
					UInt64	deviceTargetID = 0;
					
					// Get the value from the CFNumberRef.
					if ( CFNumberGetValue ( deviceTargetIDRef, kCFNumberLongLongType, &deviceTargetID ) )
					{
						
						// Does it match?
						if ( targetID == deviceTargetID )
						{
							
							// Reprobe the device.
							result = IOServiceRequestProbe ( service, 0 );
							found = true;
							
						}
						
					}
					
				}
				
			}
			
			if ( deviceDict != NULL )
				CFRelease ( deviceDict );
			
		}
		
		IOObjectRelease ( service );
		
		service = IOIteratorNext ( childIter );
		
	}
	
	IOObjectRelease ( childIter );
	childIter = MACH_PORT_NULL;
	
	if ( found == false )
		result = kIOReturnNoDevice;
	
	
ErrorExit:
	
	
	return result;
	
}