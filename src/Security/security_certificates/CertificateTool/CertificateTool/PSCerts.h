/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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
//  PSCerts.h
//  CertificateTool
//
//  Copyright (c) 2012-2013 Apple Inc. All Rights Reserved.
//

#import <Foundation/Foundation.h>

extern NSString* kSecAnchorTypeUndefined;
extern NSString* kSecAnchorTypeSystem;
extern NSString* kSecAnchorTypePlatform;
extern NSString* kSecAnchorTypeCustom;

@interface PSCerts : NSObject
{
    NSString*       _cert_dir_path;
    NSMutableArray* _certs;
    NSNumber*       _flags;
    BOOL            _recurse;
}

@property (readonly) NSArray* certs;

- (id)initWithCertFilePath:(NSString *)filePath withFlags:(NSNumber *)flags;
- (id)initWithCertFilePath:(NSString *)filePath withFlags:(NSNumber *)flags recurse:(BOOL)recurse;

@end
