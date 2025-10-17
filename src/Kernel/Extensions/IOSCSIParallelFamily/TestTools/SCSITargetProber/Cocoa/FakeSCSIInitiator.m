/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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

#import "FakeSCSIInitiator.h"

static int sDomainID = 0;


//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ
//	Implementation
//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ


@implementation FakeSCSIInitiator


- ( id ) init
{
	
	int i = 0;
	
	self = [ super init ];
	
	if ( self != nil )
	{
		
		initiatorID = 7;
		domainID	= sDomainID++;
		
		for ( i = 0; i < 16; i++ )
		{
			
			SCSIDevice *	device = [ [ SCSIDevice alloc ] init ];
			
			if ( i == initiatorID )
			{
				
				[ device setPhysicalInterconnect: @"SCSI Parallel Interface" ];
	
				if ( ( domainID & 1 ) == 1 )
				{
					[ device setManufacturer: @"LSILogic" ];
					[ device setModel: @"LSI,1030" ];
				}
				else
				{
					[ device setManufacturer: @"ATTO" ];
					[ device setModel: @"ATTO,UL4D" ];
				}
				
				[ device setImage: [ NSImage imageNamed: @"pcicard" ] ];
				[ device setRevision: @"1.52b4" ];
				[ device setDomainIdentifier: [ NSNumber numberWithInt: domainID ] ];
				[ device setDeviceIdentifier: [ NSNumber numberWithInt: i ] ];
				[ device setPeripheralDeviceType: @"N/A" ];
				[ device setFeatures: [ NSArray arrayWithObjects: @"Fast", @"Wide", nil ] ];
				[ device setIsInitiator: YES ];
				[ device setDevicePresent: YES ];
				[ devices addObject: device ];
				initiatorDevice = device;
				[ device release ];
				
			}
			
			else if ( i == 4 )
			{
				
				[ device setImage: [ NSImage imageNamed: @"nothing" ] ];
				[ device setIsInitiator: NO ];
				[ device setDevicePresent: NO ];
				[ device setDomainIdentifier: [ NSNumber numberWithInt: domainID ] ];
				[ device setDeviceIdentifier: [ NSNumber numberWithInt: i ] ];
				[ devices addObject: device ];
				[ device release ];
				
			}
			
			else
			{
				
				[ device setPhysicalInterconnect: @"SCSI Parallel Interface" ];
				
				if ( ( domainID & 1 ) == 1 )
				{
					
					[ device setManufacturer: @"Seagate" ];
					[ device setModel: @"Cheetah" ];
					[ device setRevision: @"1.0b1" ];
					
				}
				
				else
				{
					
					[ device setManufacturer: @"IBM" ];
					[ device setModel: @"SuperFastDisk" ];
					[ device setRevision: @"H641" ];
					
				}
				
				[ device setImage: [ NSImage imageNamed: @"hd" ] ];
				[ device setIsInitiator: NO ];
				[ device setDevicePresent: YES ];
				[ device setDomainIdentifier: [ NSNumber numberWithInt: domainID ] ];
				[ device setDeviceIdentifier: [ NSNumber numberWithInt: i ] ];
				[ device setPeripheralDeviceType: @"0" ];
				[ device setFeatures: [ NSArray arrayWithObjects: @"Fast", @"Wide", @"DT", @"IU", @"QAS", nil ] ];
				[ devices addObject: device ];
				[ device release ];
				
			}
			
		}
		
	}
	
	return self;
	
}


@end