/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
//  PSCert.h
//  CertificateTool
//
//  Copyright (c) 2012-2013 Apple Inc. All Rights Reserved.
//

#import <Foundation/Foundation.h>

@interface PSCert : NSObject
{
@private
    NSData*             _cert_data;
    NSNumber*			_flags;
    NSData*             _normalized_subject_hash;
    NSData*             _certificate_hash;
    NSData*             _certificate_sha256_hash;
	NSData*				_public_key_hash;
    NSData*             _spki_hash;
    NSString*           _file_path;
    NSString*           _auth_key_id;
    NSString*           _subj_key_id;
    NSString*           _anchor_type;
}

@property (readonly) NSData* cert_data;
@property (readonly) NSData* normalized_subject_hash;
@property (readonly) NSData* certificate_hash;
@property (readonly) NSData* certificate_sha256_hash;
@property (readonly) NSData* public_key_hash;
@property (readonly) NSData* spki_hash;
@property (readonly) NSString* file_path;
@property (readonly) NSString* auth_key_id;
@property (readonly) NSString* subj_key_id;
@property (readonly) NSNumber* flags;
@property (readonly) NSString* anchor_type;

- (id)initWithCertFilePath:(NSString *)filePath withFlags:(NSNumber*)flags;

@end
