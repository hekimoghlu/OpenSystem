/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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
//  PSCertKey.m
//  CertificateTool
//
//  Copyright (c) 2012-2013 Apple Inc. All Rights Reserved.
//

#import "PSCertKey.h"
#import <Security/Security.h>
#import "PSUtilities.h"

@implementation PSCertKey

@synthesize key_hash = _key_hash;


- (id)initWithCertFilePath:(NSString *)filePath
{
    if ((self = [super init]))
    {
        _key_hash = nil;
        
        CFDataRef temp_cf_data = CFBridgingRetain([PSUtilities readFile:filePath]);
        if (NULL == temp_cf_data)
        {
            NSLog(@"PSCertKey: Unable to read data for file %@", filePath);
            return nil;
        }
        
        SecCertificateRef aCert = [PSUtilities getCertificateFromData:temp_cf_data];
        CFRelease(temp_cf_data);
        if (NULL != aCert)
        {
            CFDataRef temp_key_data = [PSUtilities getKeyDataFromCertificate:aCert];
            if (NULL != temp_key_data)
            {
                _key_hash = [PSUtilities digestAndEncode:temp_key_data useSHA1:YES];
            }
            CFRelease(aCert);
        }
    }
    return self;
}

@end
