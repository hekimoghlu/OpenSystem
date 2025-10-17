/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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
#import <XCTest/XCTest.h>
#include "utilities/der_plist.h"
#include "SecCFWrappers.h"

@interface testPlistDER : XCTestCase
@end

static CFDataRef CreateDERFromDictionary(CFDictionaryRef di, CFErrorRef *error)
{
    size_t size = der_sizeof_plist(di, error);
    if (size == 0)
        return NULL;
    uint8_t *der = malloc(size);
    if (der == NULL) {
        return NULL;
    }
    der_encode_plist(di, error, der, der+size);
    return CFDataCreateWithBytesNoCopy(kCFAllocatorDefault, der, size, kCFAllocatorMalloc);
}

@implementation testPlistDER


- (void)testSecPListLargeData {
    NSMutableData *data = [NSMutableData dataWithLength:650000];
    memset([data mutableBytes], 'A', [data length]);

    NSDictionary *dictionary = @{
        @"BackupKey" : [NSMutableData dataWithLength:32],
        @"DeviceID" : data,
        @"EscrowRecord" : @"<null>",
        @"PreferIDFragmentation" : @(1),
        @"PreferIDS" : @(0),
        @"PreferIDSAckModel" : @(1),
        @"SecurityProperties" : @{},
        @"SerialNumber" : @"C02TD01QHXCW",
        @"TransportType" : @"KVS",
        @"Views" : @[
                @"iCloudIdentity",
                @"BackupBagV0",
                @"PCS-Maildrop",
                @"PCS-iMessage",
                @"PCS-Notes",
                @"PCS-FDE",
                @"PCS-MasterKey",
                @"NanoRegistry",
                @"PCS-Feldspar",
                @"PCS-iCloudDrive",
                @"AccessoryPairing",
                @"ContinuityUnlock",
                @"WatchMigration",
                @"PCS-Sharing",
                @"PCS-Photos",
                @"PCS-Escrow",
                @"AppleTV",
                @"HomeKit",
                @"PCS-Backup",
                @"PCS-CloudKit"
        ],
    };
    CFErrorRef error = NULL;

    size_t size = der_sizeof_plist((__bridge CFTypeRef)dictionary, &error);
    XCTAssertNotEqual(size, (size_t)0, "no data?: %@", error);
    CFReleaseNull(error);

    uint8_t *der = malloc(size);
    uint8_t *der_end = der + size;
    uint8_t *der_fin = der_encode_plist((__bridge CFTypeRef)dictionary, &error, der, der_end);

    XCTAssert(error == NULL, "error should be NULL: %@", error);
    XCTAssertEqual(der, der_fin, "under/over-flow");

    free(der);

    CFReleaseNull(error);

    NSData *outdata = (__bridge_transfer NSData *)CreateDERFromDictionary((__bridge CFTypeRef)dictionary, &error);
    XCTAssertEqual(error, NULL, "error should be NULL: %@", error);
    XCTAssertNotEqual(outdata, NULL, "should have data");

}

- (void)testSecPListLargeDataOtherThread
{
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        [self testSecPListLargeData];
        dispatch_semaphore_signal(sema);
    });
    dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
}


@end
