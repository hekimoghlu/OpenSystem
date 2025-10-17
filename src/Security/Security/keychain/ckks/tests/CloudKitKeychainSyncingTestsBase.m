/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
#if OCTAGON

#import "CloudKitKeychainSyncingTestsBase.h"

@implementation CloudKitKeychainSyncingTestsBase

- (ZoneKeys*)keychainZoneKeys {
    return self.keys[self.keychainZoneID];
}

- (BOOL)mockPostFollowUpWithContext:(CDPFollowUpContext *)context error:(NSError **)error {
    secnotice("octagon", "mock cdp posting follow up");
    return YES;
}

// Override our base class
- (NSSet<NSString*>*)managedViewList {
    return [NSSet setWithObject:@"keychain"];
}

+ (void)setUp {
    SecCKKSEnable();
    SecCKKSResetSyncing();
    KCSharingSetChangeTrackingEnabled(false);

    [super setUp];
}

- (void)setUp {
    self.utcCalendar = [NSCalendar calendarWithIdentifier:NSCalendarIdentifierISO8601];
    self.utcCalendar.timeZone = [NSTimeZone timeZoneWithAbbreviation:@"UTC"];

    [super setUp];

    if(SecCKKSIsEnabled()) {
        self.keychainZoneID = [[CKRecordZoneID alloc] initWithZoneName:@"keychain" ownerName:CKCurrentUserDefaultName];
        self.keychainZone = [[FakeCKZone alloc] initZone: self.keychainZoneID];

        [self.ckksZones addObject:self.keychainZoneID];

        // Wait for the ViewManager to be brought up
        XCTAssertEqual(0, [self.injectedManager.completedSecCKKSInitialize wait:20*NSEC_PER_SEC], "No timeout waiting for SecCKKSInitialize");

        self.defaultCKKS = [[CKKSViewManager manager] ckksAccountSyncForContainer:SecCKKSContainerName
                                                                        contextID:OTDefaultContext];

        self.keychainView = [self.defaultCKKS.operationDependencies viewStateForName:@"keychain"];
        XCTAssertNotNil(self.keychainView, "CKKS knows about the keychain view");
        [self.ckksViews addObject:self.keychainView];
    }
}


+ (void)tearDown {
    [super tearDown];
    KCSharingClearChangeTrackingEnabledOverride();
    SecCKKSResetSyncing();
}

- (void)tearDown {
    // Fetch status, to make sure we can

    // Can only fetch status from the default persona.
    [self.mockPersonaAdapter prepareThreadForKeychainAPIUseForPersonaIdentifier:nil];
    self.mockPersonaAdapter.isDefaultPersona = YES;

    XCTestExpectation* statusCompletes = [self expectationWithDescription:@"status completes"];
    [self.defaultCKKS rpcStatus:nil
                      fast:NO
                      waitForNonTransientState:CKKSControlStatusDefaultNonTransientStateTimeout
                      reply:^(NSArray<NSDictionary*>* _Nullable status, NSError* _Nullable error) {
        XCTAssertNotNil(status, "Should have some statuses");
        XCTAssertNil(error, "Should have no error fetching status");
        [statusCompletes fulfill];
    }];
    [self waitForExpectations:@[statusCompletes] timeout:20];

    [self.defaultCKKS halt];
    [self.defaultCKKS waitUntilAllOperationsAreFinished];

    self.keychainView = nil;
    self.keychainZoneID = nil;

    [super tearDown];
} 

- (FakeCKZone*)keychainZone {
    return self.zones[self.keychainZoneID];
}

- (void)setKeychainZone: (FakeCKZone*) zone {
    self.zones[self.keychainZoneID] = zone;
}

@end

#endif /* OCTAGON */
