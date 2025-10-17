/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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

#import <Cocoa/Cocoa.h>
#import <IOKit/IOKitLib.h>


//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ
//	Interface
//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ

@interface SCSIDevice : NSObject
{
	NSString *		physicalInterconnect;
	NSString *		manufacturer;
	NSString *		model;
	NSString *		revision;
	NSString *		peripheralDeviceType;
	NSNumber *		domainIdentifier;
	NSNumber *		deviceIdentifier;
	NSArray *		features;
	NSImage *		image;
	BOOL			devicePresent;
	BOOL			isInitiator;
}

- ( id ) initWithService: ( io_service_t ) service;
+ ( int ) domainIDForService: ( io_service_t ) service;
+ ( int ) targetIDForService: ( io_service_t ) service;
- ( void ) setPhysicalInterconnect: ( NSString * ) p;
- ( void ) setManufacturer: ( NSString * ) m;
- ( void ) setModel: ( NSString * ) m;
- ( void ) setRevision: ( NSString * ) r;
- ( void ) setPeripheralDeviceType: ( NSString * ) p;
- ( void ) setDomainIdentifier: ( NSNumber * ) i;
- ( void ) setDeviceIdentifier: ( NSNumber * ) i;
- ( void ) setFeatures: ( NSArray * ) f;
- ( void ) setIsInitiator: ( BOOL ) value;
- ( void ) setDevicePresent: ( BOOL ) value;
- ( void ) setImage: ( NSImage * ) image;

- ( NSString * ) title;
- ( NSString * ) physicalInterconnect;
- ( NSString * ) manufacturer;
- ( NSString * ) model;
- ( NSString * ) revision;
- ( NSString * ) peripheralDeviceType;
- ( NSNumber * ) domainIdentifier;
- ( NSNumber * ) deviceIdentifier;
- ( NSArray * ) features;
- ( NSImage * ) image;
- ( BOOL ) devicePresent;
- ( BOOL ) isInitiator;

- ( IBAction ) reprobe: ( id ) sender;

- ( void ) clearInformation;
- ( NSArray * ) buildFeatureList: ( NSNumber * ) number;


@end