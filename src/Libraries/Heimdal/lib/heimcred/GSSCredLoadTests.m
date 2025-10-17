/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import "GSSCredTestUtil.h"
#import <XCTest/XCTest.h>
#import "gsscred.h"
#import "hc_err.h"
#import "common.h"
#import "heimbase.h"
#import "heimcred.h"
#import "mock_aks.h"
#import "acquirecred.h"

@interface GSSCredLoadTests : XCTestCase
@property (nullable) struct peer * peer;
@property (nonatomic) MockManagedAppManager *mockManagedAppManager;
@end

@implementation GSSCredLoadTests {
}
@synthesize peer;
@synthesize mockManagedAppManager;

- (void)setUp {

    self.mockManagedAppManager = [[MockManagedAppManager alloc] init];

    HeimCredGlobalCTX.isMultiUser = NO;
    HeimCredGlobalCTX.currentAltDSID = currentAltDSIDMock;
    HeimCredGlobalCTX.hasEntitlement = haveBooleanEntitlementMock;
    HeimCredGlobalCTX.getUid = getUidMock;
    HeimCredGlobalCTX.getAsid = getAsidMock;
    HeimCredGlobalCTX.encryptData = encryptDataMock;
    HeimCredGlobalCTX.decryptData = decryptDataMock;
    HeimCredGlobalCTX.managedAppManager = self.mockManagedAppManager;
    HeimCredGlobalCTX.useUidMatching = NO;
    HeimCredGlobalCTX.disableNTLMReflectionDetection = NO;
    HeimCredGlobalCTX.verifyAppleSigned = verifyAppleSignedMock;
    HeimCredGlobalCTX.sessionExists = sessionExistsMock;
    HeimCredGlobalCTX.saveToDiskIfNeeded = saveToDiskIfNeededMock;
    HeimCredGlobalCTX.getValueFromPreferences = getValueFromPreferencesMock;
    HeimCredGlobalCTX.expireFunction = expire_func;
    HeimCredGlobalCTX.renewFunction = renew_func;
    HeimCredGlobalCTX.finalFunction = final_func;
    HeimCredGlobalCTX.notifyCaches = NULL;
    HeimCredGlobalCTX.gssCredHelperClientClass = nil;

    CFRELEASE_NULL(HeimCredCTX.mechanisms);
    HeimCredCTX.mechanisms = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    heim_assert(HeimCredCTX.mechanisms != NULL, "out of memory");

    CFRELEASE_NULL(HeimCredCTX.schemas);
    HeimCredCTX.schemas = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    heim_assert(HeimCredCTX.schemas != NULL, "out of memory");

    HeimCredCTX.globalSchema = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    heim_assert(HeimCredCTX.globalSchema != NULL, "out of memory");

    _HeimCredRegisterGeneric();
    _HeimCredRegisterConfiguration();
    _HeimCredRegisterKerberos();
    _HeimCredRegisterKerberosAcquireCred();
    _HeimCredRegisterNTLM();
    _HeimCredRegisterNTLMReflection();

    CFRELEASE_NULL(HeimCredCTX.globalSchema);

#if TARGET_OS_SIMULATOR
    archivePath = [[NSString alloc] initWithFormat:@"%@/Library/Caches/com.apple.GSSCred.simulator-archive.test", NSHomeDirectory()];
#else
    archivePath = @"/var/tmp/heim-credential-store.archive.test";
#endif
    _HeimCredInitCommon();
    CFRELEASE_NULL(HeimCredCTX.sessions);
    HeimCredCTX.sessions = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

    CFRELEASE_NULL(HeimCredCTX.challenges);
    HeimCredCTX.challenges = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    //always start clean
    NSError *error;
    [[NSFileManager defaultManager] removeItemAtPath:archivePath error:&error];

    //default test values
    _entitlements = @[];
    _currentUid = 501;
    _altDSID = NULL;
    _currentAsid = 10000;
}

- (void)tearDown {

    NSError *error;
    [[NSFileManager defaultManager] removeItemAtPath:archivePath error:&error];
    [GSSCredTestUtil freePeer:self.peer];
    self.peer = NULL;

    CFRELEASE_NULL(HeimCredCTX.sessions);
    HeimCredCTX.sessions = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
}

//pragma mark - Tests

//add credential and fetch it
- (void)testLoadingTooLargeFile {


    long sizeToMake = 1024*1024*6;
    NSMutableData *fakeData = [NSMutableData dataWithCapacity:sizeToMake];

    for (long i=0; i<sizeToMake; i=i+4) {
	u_int32_t random = arc4random();
	[fakeData appendBytes:(void*)&random length:4];
    }

    [fakeData writeToFile:archivePath atomically:NO];

    cache_read_status result = readCredCache();

    XCTAssertEqual(result, READ_SIZE_ERROR, "An 6 meg file should not load on macOS");

}

- (void)testLoadingNotTooLargeFile {


    long sizeToMake = 1024*6;
    NSMutableData *fakeData = [NSMutableData dataWithCapacity:sizeToMake];

    for (long i=0; i<sizeToMake; i=i+4) {
	u_int32_t random = arc4random();
	[fakeData appendBytes:(void*)&random length:4];
    }

    [fakeData writeToFile:archivePath atomically:NO];

    cache_read_status result = readCredCache();

    XCTAssertEqual(result, READ_EMPTY, "An 600 Kb file should not load on macOS");

}

// mocks

static NSArray<NSString*> *_entitlements;
static NSString *_altDSID;
static int _currentUid;
static int _currentAsid;
static NSString *_currentSignedIdentifier;

static NSString * currentAltDSIDMock(void)
{
    return _altDSID;
}

static bool haveBooleanEntitlementMock(struct peer *peer, const char *entitlement)
{
    NSString *ent = @(entitlement);
    return [_entitlements containsObject:ent];
}

static bool verifyAppleSignedMock(struct peer *peer, NSString *identifer)
{
    return ([identifer isEqualToString:_currentSignedIdentifier]);
}

static bool sessionExistsMock(pid_t asid) {
    return true;
}

//xpc mock

static uid_t getUidMock(xpc_connection_t connection) {
    return _currentUid;
}

static au_asid_t getAsidMock(xpc_connection_t connection) {
    return _currentAsid;
}

static void saveToDiskIfNeededMock(void)
{

}

static CFPropertyListRef getValueFromPreferencesMock(CFStringRef key)
{
    return NULL;
}
@end

