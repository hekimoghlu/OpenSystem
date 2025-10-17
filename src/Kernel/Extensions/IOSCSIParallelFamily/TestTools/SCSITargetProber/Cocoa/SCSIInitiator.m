/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ
//	Imports
//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ

#import "SCSIInitiator.h"
#import "SCSITargetProberKeys.h"
#import <IOKit/IOKitLib.h>
#import <IOKit/storage/IOStorageProtocolCharacteristics.h>


//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ
//	Constants
//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ

#define kSCSIInitiatorIdentifierString		"SCSI Initiator Identifier"
#define kSocketNumberString					"Socket Number"

#define kPCIKeyModel						"model"
#define kPCIKeyVendorID						"vendor-id"
#define kPCIKeyVersion						"version"
#define kPCIKeyName							"name"


//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ
//	Prototypes
//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ

static int
deviceComparator ( id obj1, id obj2, void * context );


//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ
//	Implementation
//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ

@implementation SCSIInitiator


- ( NSArray * ) devices { return devices; }


- ( void ) dealloc
{
	
	[ self setInitiatorDevice: nil ];
	[ self setDevices: nil ];
	
}


- ( void ) setInitiatorDevice: ( SCSIDevice * ) i
{
	
	[ i retain ];
	[ initiatorDevice release ];
	initiatorDevice = i;
	
}


- ( void ) setDevices: ( NSMutableArray * ) d
{
	
	[ d retain ];
	[ devices release ];
	devices = d;
	
}


- ( int ) initiatorID
{
	return initiatorID;
}


- ( int ) domainID
{
	return domainID;
}


- ( NSString * ) title
{
	
	NSString *	string = @"No Controllers Found";
	
	if ( initiatorDevice != nil )
	{
		string = [ NSString stringWithFormat: @"%d:  %@", domainID, [ initiatorDevice title ]];
	}
	
	return string;
	
}


- ( id ) init
{
	
	self = [ super init ];
	
	if ( self != nil )
	{
		
		[ self setDevices: [ [ [ NSMutableArray alloc ] init ] autorelease ] ];
		[ self setInitiatorDevice: nil ];
		
	}
	
	return self;
	
}


- ( id ) initWithService: ( io_service_t ) service
{
	
	self = [ super init ];
	
	if ( self != nil )
	{
		
		[ self setDevices: [ [ [ NSMutableArray alloc ] init ] autorelease ] ];
		[ self setInitiatorDevice: nil ];
		[ self setInitiatorProperties: service ];
		[ self setTargetProperties: service ];
		
	}
	
	return self;
	
}


// Called to get the SCSI Domain ID for the io_service_t.

+ ( int ) domainIDForService: ( io_service_t ) service
{
	
	int			domain	= 0;
	id			value	= nil;

	// Get the protocol characteristics key.
	value = ( id ) IORegistryEntrySearchCFProperty ( service,
													 kIOServicePlane,
													 CFSTR ( kIOPropertyProtocolCharacteristicsKey ),
													 kCFAllocatorDefault,
													 0 );
	
	// Get the SCSI Domain ID.
	domain = [ [ value objectForKey: @kIOPropertySCSIDomainIdentifierKey ] intValue ];
	[ value release ];
	
	return domain;
	
}


// Called to add a target device to the devices array.

- ( void ) addTargetDevice: ( SCSIDevice * ) newDevice
{
	
	int				count   = 0;
	int				index   = 0;
	SCSIDevice *	device  = nil;
	
	count = [ devices count ];
	
	// Broadcast to any listeners that we are updating the devices array.
	[ self willChangeValueForKey: kDevicesKeyPath ];
	
	for ( index = 0; index < count; index++ )
	{
		
		device = [ devices objectAtIndex: index ];
		
		// Is this the correct device?
		if ( [ [ device deviceIdentifier ] intValue ] == [ [ newDevice deviceIdentifier ] intValue ] )
		{
			
			// Yes, replace the device at this location with the updated one.
			[ devices replaceObjectAtIndex: index withObject: newDevice ];
			break;
			
		}
		
	}
	
	// Broadcast to any listeners that we finished updating the devices array.
	[ self didChangeValueForKey: kDevicesKeyPath ];
	
}


- ( void ) removeTargetDevice: ( int ) targetID
{
	
	int				count   = 0;
	int				index   = 0;
	SCSIDevice *	device  = nil;
	
	count = [ devices count ];
	
	// Broadcast to any listeners that we are updating the devices array.	
	[ self willChangeValueForKey: kDevicesKeyPath ];
	
	for ( index = 0; index < count; index++ )
	{
		
		// Is this the correct device?
		device = [ devices objectAtIndex: index ];
		if ( [ [ device deviceIdentifier ] intValue ] == targetID )
		{
			
			// Yes, force the information to get cleared since the
			// target no longer exists.
			[ device clearInformation ];
			
		}
		
	}
	
	// Broadcast to any listeners that we finished updating the devices array.
	[ self didChangeValueForKey: kDevicesKeyPath ];
	
}


// Called to set the properties up for the initiator.

- ( void ) setInitiatorProperties: ( io_service_t ) service
{
	
	id	value = nil;
	
	// Create the device which represents the initiator.
	initiatorDevice = [ [ SCSIDevice alloc ] init ];
	
	// Set some initial defaults.
	[ initiatorDevice setDevicePresent: YES ];
	[ initiatorDevice setIsInitiator: YES ];
	
	// Find the initiator identifier.
	value = ( id ) IORegistryEntrySearchCFProperty ( service,
													 kIOServicePlane,
													 CFSTR ( kSCSIInitiatorIdentifierString ),
													 kCFAllocatorDefault,
													 0 );
	
	initiatorID = [ value intValue ];
	[ value release ];
	
	// Find the SCSI Domain Identifier.
	value = ( id ) IORegistryEntrySearchCFProperty ( service,
													 kIOServicePlane,
													 CFSTR ( kIOPropertyProtocolCharacteristicsKey ),
													 kCFAllocatorDefault,
													 0 );
	
	domainID = [ [ value objectForKey: @kIOPropertySCSIDomainIdentifierKey ] intValue ];
	
	// Set the physical interconnect.
	[ initiatorDevice setPhysicalInterconnect: [ value objectForKey: @kIOPropertyPhysicalInterconnectTypeKey ] ];
	[ value release ];
	
	// Set the domainID and deviceID for the initiator device.
	[ initiatorDevice setDomainIdentifier: [ NSNumber numberWithInt: domainID ] ];
	[ initiatorDevice setDeviceIdentifier: [ NSNumber numberWithInt: initiatorID ] ];
	
	// Cheesy hack to find out if this is Card Bus or PCI,
	// need a better way to determine this in the future.
	value = ( id ) IORegistryEntrySearchCFProperty ( service,
													 kIOServicePlane,
													 CFSTR ( kSocketNumberString ),
													 kCFAllocatorDefault,
													 kIORegistryIterateRecursively | kIORegistryIterateParents );
	if ( value == nil )
		[ initiatorDevice setImage: [ NSImage imageNamed: kPCICardImageString ] ];
	else
		[ initiatorDevice setImage: [ NSImage imageNamed: kPCCardImageString ] ];
	
	// The card doesn't have any features. Maybe we can add some later...
	[ initiatorDevice setFeatures: nil ];
	
	// Not all PCI/CardBus devices have these keys. Need to look at more
	// cards and see if we can glean more information for things like Firmware Revision,
	// FCode revision, etc. Maybe in the next release...
	value = ( id ) IORegistryEntrySearchCFProperty ( service,
													 kIOServicePlane,
													 CFSTR ( kPCIKeyVersion ),
													 kCFAllocatorDefault,
													 kIORegistryIterateRecursively | kIORegistryIterateParents );
	
	[ initiatorDevice setRevision: [ NSString stringWithCString: value ? [ value bytes ] : "Unknown" ] ];

	value = ( id ) IORegistryEntrySearchCFProperty ( service,
													 kIOServicePlane,
													 CFSTR ( kPCIKeyModel ),
													 kCFAllocatorDefault,
													 kIORegistryIterateRecursively | kIORegistryIterateParents );
	
	[ initiatorDevice setModel: [ NSString stringWithCString: value ? [ value bytes ] : "Unknown" ] ];
	
	value = ( id ) IORegistryEntrySearchCFProperty ( service,
													 kIOServicePlane,
													 CFSTR ( kPCIKeyName ),
													 kCFAllocatorDefault,
													 kIORegistryIterateRecursively | kIORegistryIterateParents );
	[ initiatorDevice setManufacturer: [ NSString stringWithCString: value ? [ value bytes ] : "Unknown" ] ];
	
	// Add the initiator device to the list of devices.
	[ devices addObject: initiatorDevice ];
	
}


// Called to set the properties up for the target devices.

- ( void ) setTargetProperties: ( io_service_t ) service
{
	
	IOReturn			result		= kIOReturnSuccess;
	io_iterator_t		iterator	= MACH_PORT_NULL;
	SCSIDevice *		device		= nil;
	
	// Get a child iterator.
	result = IORegistryEntryGetChildIterator ( service,
											   kIOServicePlane,
											   &iterator );
	
	if ( result == kIOReturnSuccess )
	{
		
		io_service_t	child = MACH_PORT_NULL;
		
		child = IOIteratorNext ( iterator );
		
		while ( child != MACH_PORT_NULL )
		{
			
			// Create a SCSIDevice object to represent this device.
			device = [ [ SCSIDevice alloc ] initWithService: child ];
			
			// Add the device to the list.
			[ devices addObject: device ];
			
			// Proceed to the next...
			IOObjectRelease ( child );
			child = IOIteratorNext ( iterator );
			
		}
		
		IOObjectRelease ( iterator );
		iterator = MACH_PORT_NULL;
		
	}
	
	// Sort the devices we found. The sorting function uses the
	// targetID to sort. This is arbitrary, but works well in this
	// case since we display stuff from lowest targetID to highest.
	[ devices sortUsingFunction: deviceComparator context: nil ];
	
}


#if 0
#pragma mark -
#pragma mark Static Methods
#pragma mark -
#endif


//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ
//	deviceComparator - Compares devices based on deviceIdentifier field
//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ

int
deviceComparator ( id obj1, id obj2, void * context )
{

#pragma unused ( context )
	
	int		result = NSOrderedSame;
	
	if ( [ [ obj1 deviceIdentifier ] intValue ] < [ [ obj2 deviceIdentifier ] intValue ] )
	{
		result = NSOrderedAscending;
	}
	
	else if ( [ [ obj1 deviceIdentifier ] intValue ] > [ [ obj2 deviceIdentifier ] intValue ] )
	{
		result = NSOrderedDescending;
	}
	
	return result;
	
}


@end