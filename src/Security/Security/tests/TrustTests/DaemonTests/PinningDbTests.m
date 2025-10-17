/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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
//  PinningDbTests.m
//

#import <XCTest/XCTest.h>
#import <Foundation/Foundation.h>
#import <sqlite3.h>
#import "trust/trustd/SecPinningDb.h"

#import "TrustDaemonTestCase.h"

@interface PinningDbInitializationTests : TrustDaemonInitializationTestCase
@end

@implementation PinningDbInitializationTests

- (void)testSchemaUpgrade
{
#if TARGET_OS_BRIDGE
    /* BridgeOS doesn't have security_certificates project so there is no baseline pinning plist */
    XCTSkip();
#endif
    /* Create a "pinningDB" with a large content version number but older schema version */
    char *schema_v2 =   "PRAGMA foreign_keys=OFF; "
                        "PRAGMA user_version=2; "
                        "BEGIN TRANSACTION; "
                        "CREATE TABLE admin(key TEXT PRIMARY KEY NOT NULL,ival INTEGER NOT NULL,value BLOB); "
                        "INSERT INTO admin VALUES('version',2147483647,NULL); " // Version as INT_MAX
                        "CREATE TABLE rules( policyName TEXT NOT NULL,"
                                            "domainSuffix TEXT NOT NULL,"
                                            "labelRegex TEXT NOT NULL,"
                                            "policies BLOB NOT NULL,"
                                            "UNIQUE(policyName, domainSuffix, labelRegex)); "
                        "COMMIT;";
    NSURL *pinningDbPath = [SecPinningDb pinningDbPath];
    sqlite3 *handle = nil;
    int sqlite_result = sqlite3_open_v2([pinningDbPath fileSystemRepresentation], &handle,
                                        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL);
    XCTAssertEqual(sqlite_result, SQLITE_OK);
    XCTAssert(SQLITE_OK == sqlite3_exec(handle, schema_v2, NULL, NULL, NULL));
    XCTAssert(SQLITE_OK == sqlite3_close(handle));

    /* Initialize the Pinning DB -- schema should get upgraded and populated with system content version */
    SecPinningDb *pinningDb = [[SecPinningDb alloc] init];
    XCTAssert(SecDbPerformRead([pinningDb db], NULL, ^(SecDbConnectionRef dbconn) {
        NSNumber *contentVersion = [pinningDb getContentVersion:dbconn error:NULL];
        NSNumber *schemaVersion = [pinningDb getSchemaVersion:dbconn error:NULL];
        XCTAssert([contentVersion intValue] < INT_MAX && [contentVersion intValue] > 0);
        XCTAssertEqualObjects(schemaVersion, @(PinningDbSchemaVersion));
    }));
}

- (void)testContentUpgradeFromFile
{
#if TARGET_OS_BRIDGE
    XCTSkip();
#endif
    /* initialize a DB with the system content version */
    SecPinningDb *pinningDb = [[SecPinningDb alloc] init];
    XCTAssert(SecDbPerformRead([pinningDb db], NULL, ^(SecDbConnectionRef dbconn) {
        NSNumber *contentVersion = [pinningDb getContentVersion:dbconn error:NULL];
        XCTAssert([contentVersion intValue] < INT_MAX && [contentVersion intValue] > 0);
    }));

    /* update it using a test plist with INT_MAX version */
    NSURL *pinningPlist = [[NSBundle bundleForClass:[self class]] URLForResource:@"PinningDB_vINT_MAX" withExtension:nil
                                                                 subdirectory:@"TestTrustdInitialization-data"];
    XCTAssert([pinningDb installDbFromURL:pinningPlist error:nil]);
    XCTAssert(SecDbPerformRead([pinningDb db], NULL, ^(SecDbConnectionRef dbconn) {
        NSNumber *contentVersion = [pinningDb getContentVersion:dbconn error:NULL];
        XCTAssertEqual([contentVersion intValue], INT_MAX);
    }));

    /* update one more time with the same content version */
    XCTAssert([pinningDb installDbFromURL:pinningPlist error:nil]);
}
@end
