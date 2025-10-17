/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 13, 2025.
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
//  CKKSPBFileStorage.m
//

#import "keychain/ckks/CKKSPBFileStorage.h"

@interface CKKSPBFileStorage ()
@property NSURL *storageFile;
@property Class<CKKSPBCodable> storageClass;
@property id<CKKSPBCodable> protobufStorage;
@end

@implementation CKKSPBFileStorage

- (CKKSPBFileStorage *)initWithStoragePath:(NSURL *)storageFile
                              storageClass:(Class<CKKSPBCodable>) storageClass
{
    if ((self = [super init]) == nil) {
        return nil;
    }
    self.storageFile = storageFile;
    self.storageClass = storageClass;

    NSData *data = [NSData dataWithContentsOfURL:storageFile];
    if (data != nil) {
        self.protobufStorage = [[self.storageClass alloc] initWithData:data];
    }
    /* if not storage, or storage is corrupted, this function will return a empty storage */
    if (self.protobufStorage == nil) {
        self.protobufStorage = [[self.storageClass alloc] init];
    }

    return self;
}

- (id _Nullable)storage
{
    __block id storage;
    @synchronized (self) {
        storage = self.protobufStorage;
    }
    return storage;
}

- (void)setStorage:(id _Nonnull)storage
{
    @synchronized (self) {
        id<CKKSPBCodable> c = storage;
        NSData *data = c.data;
        [data writeToURL:self.storageFile atomically:YES];
        self.protobufStorage = [[self.storageClass alloc] initWithData:data];
    }
}


@end
