/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
//  PSCerts.m
//  CertificateTool
//
//  Copyright (c) 2012-2015 Apple Inc. All Rights Reserved.
//

#import "PSCerts.h"
#import "PSCertKey.h"
#import "PSCert.h"

@interface PSCerts (PrivateMethods)

- (BOOL)get_certs;

@end

@implementation PSCerts

@synthesize certs = _certs;

- (BOOL)get_certs
{
    BOOL result = NO;
    if (nil != _cert_dir_path)
    {

        NSFileManager* fileManager = [NSFileManager defaultManager];
        BOOL isDir = NO;
        if (![fileManager fileExistsAtPath:_cert_dir_path isDirectory:&isDir] || !isDir)
        {
            return result;
        }

        NSDirectoryEnumerator* enumer = [fileManager enumeratorAtPath:_cert_dir_path];
        if (nil == enumer)
        {
            return result;
        }

        for(NSString* cert_path_str in enumer)
        {
            if ([cert_path_str hasPrefix:@"."])
            {
                continue;
            }

            //NSLog(@"Processing file %@", cert_path_str);

            NSString* full_path = [_cert_dir_path stringByAppendingPathComponent:cert_path_str];

            if ([fileManager fileExistsAtPath:full_path isDirectory:&isDir] && isDir) {
                if (!_recurse) {
                    [enumer skipDescendants];
                }
                continue;
            }

            PSCert* aCert = [[PSCert alloc] initWithCertFilePath:full_path withFlags:_flags];
            if (nil != aCert)
            {
                [_certs addObject:aCert];
            }
        }
        result = YES;
    }
    return result;
}

- (id)initWithCertFilePath:(NSString *)filePath withFlags:(NSNumber *)flags recurse:(BOOL)recurse
{
    if (self = [super init])
    {
        _cert_dir_path = filePath;
        _flags = flags;
        _recurse = recurse;
        _certs = [NSMutableArray array];
        if (![self get_certs])
        {
            NSLog(@"Could not get certificates for path %@", filePath);
            self = nil;
        }
            
    }
    return self;
}

- (id)initWithCertFilePath:(NSString *)filePath withFlags:(NSNumber *)flags
{
    return [self initWithCertFilePath:filePath withFlags:flags recurse:YES];
}


@end
