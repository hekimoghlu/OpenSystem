/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 7, 2024.
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

//
//  PSCertRecord.m
//  CertificateTool
//
//  Created by James Murphy on 12/19/12.
//  Copyright (c) 2012 James Murphy. All rights reserved.
//

#import "PSCertRecord.h"

@interface PSCertRecord (PrivateMethod)

- (BOOL)ceateCertRecord:(NSData *)cert_data withFlags:(NSNumber *)flags

@end

@implementation PSCertRecord


- (id)initWithCertData:(NSData *)cert_data withFlags:(NSNumber *)flags
{
	if ((self = [super init]))
	{
		_cert_record = nil;
		if (![self ceateCertRecord:cert_data withFlags:flags])
		{
			NSLog(@"Could not create the certificate record");
			_cert_record = nil;
		}
	}
	return self;
}

- (BOOL)ceateCertRecord:(NSData *)cert_data withFlags:(NSNumber *)flags
{
	BOOL result = NO;

	if (nil == cert_data)
	{
		return result;
	}

	UInt32 flag_value = 0;
	if (nil != flags)
	{
		flag_value = (UInt32)[flags unsignedIntValue];
	}

	


}

@end