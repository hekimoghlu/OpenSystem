/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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
//  CKKSPBFileStorageTests.m
//

#import <XCTest/XCTest.h>
#import <Foundation/Foundation.h>

#import "keychain/ckks/CKKSPBFileStorage.h"
#import "keychain/ckks/proto/generated_source/CKKSSerializedKey.h"

@interface CKKSPBFileStorageTests : XCTestCase
@property NSURL * tempDir;
@end

@implementation CKKSPBFileStorageTests

- (void)setUp {
    self.tempDir = [[[NSFileManager defaultManager] temporaryDirectory] URLByAppendingPathComponent:[[NSUUID UUID] UUIDString]];
    [[NSFileManager defaultManager] createDirectoryAtURL:self.tempDir withIntermediateDirectories:NO attributes:nil error:nil];
}
- (void)tearDown {
    [[NSFileManager defaultManager] removeItemAtURL:self.tempDir error:nil];
    self.tempDir = nil;
}

- (void)testCKKSPBStorage {

    NSURL *file = [self.tempDir URLByAppendingPathComponent:@"file"];

    CKKSPBFileStorage<CKKSSerializedKey *> *pbstorage;

    pbstorage = [[CKKSPBFileStorage alloc] initWithStoragePath:file
                                                  storageClass:[CKKSSerializedKey class]];
    XCTAssertNotNil(pbstorage, "CKKSPBFileStorage should create an object");

    CKKSSerializedKey *storage = pbstorage.storage;
    storage.uuid = @"uuid";
    storage.zoneName = @"uuid";
    storage.keyclass = @"ak";
    storage.key = [NSData data];

    [pbstorage setStorage:storage];

    pbstorage = [[CKKSPBFileStorage alloc] initWithStoragePath:file
                                                  storageClass:[CKKSSerializedKey class]];
    XCTAssertNotNil(pbstorage, "CKKSPBFileStorage should create an object");

    XCTAssertEqualObjects(pbstorage.storage.keyclass, @"ak", "should be the same");
}

@end
