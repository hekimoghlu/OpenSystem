/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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
#import "aks.h"
#import "mock_aks.h"
#import "acquirecred.h"

@interface GSSCredHoldTests : XCTestCase
@property (nullable) struct peer * peer;
@property (nonatomic) MockManagedAppManager *mockManagedAppManager;
@end

@implementation GSSCredHoldTests {
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
    HeimCredGlobalCTX.encryptData = ksEncryptData;
    HeimCredGlobalCTX.decryptData = ksDecryptData;
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
    archivePath = [[NSString alloc] initWithFormat:@"%@/Library/Caches/com.apple.GSSCred.simulator-archive", NSHomeDirectory()];
#else
    archivePath = @"/var/tmp/heim-credential-store.archive";
#endif
    _HeimCredInitCommon();
    CFRELEASE_NULL(HeimCredCTX.sessions);
    HeimCredCTX.sessions = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

    CFRELEASE_NULL(HeimCredCTX.challenges);
    HeimCredCTX.challenges = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    
    //always start clean
    NSError *error;
    [[NSFileManager defaultManager] removeItemAtPath:archivePath error:&error];

    readCredCache();

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
- (void)testCreatingAndHoldingCredential {
    HeimCredGlobalCTX.isMultiUser = NO;
    HeimCredGlobalCTX.useUidMatching = NO;
    [GSSCredTestUtil freePeer:self.peer];
    self.peer = [GSSCredTestUtil createPeer:@"com.apple.fake" identifier:0];

    CFUUIDRef uuid = NULL;
    BOOL worked = [GSSCredTestUtil createCredentialAndCache:self.peer name:@"test@EXAMPLE.COM" returningCacheUuid:&uuid];

    XCTAssertTrue(worked, "Credential should be created successfully");

    CFDictionaryRef attributes;
    worked = [GSSCredTestUtil fetchCredential:self.peer uuid:uuid returningDictionary:&attributes];
    XCTAssertTrue(worked, "Cache should be fetched successfully using it's uuid");

    int64_t error = [GSSCredTestUtil hold:self.peer uuid:uuid];
    XCTAssertEqual(error, 0, "hold should not error");

    CFRELEASE_NULL(attributes);
    worked = [GSSCredTestUtil fetchCredential:self.peer uuid:uuid returningDictionary:&attributes];
    XCTAssertTrue(worked, "Cache should still exist after hold");

    error = [GSSCredTestUtil unhold:self.peer uuid:uuid];
    XCTAssertEqual(error, 0, "unhold should not error");

    error = [GSSCredTestUtil unhold:self.peer uuid:uuid];
    XCTAssertEqual(error, 0, "hold when it should delete should not error");

    CFRELEASE_NULL(attributes);
    worked = [GSSCredTestUtil fetchCredential:self.peer uuid:uuid returningDictionary:&attributes];
    XCTAssertFalse(worked, "Cache should be deleted after retain count is zero");

    CFRELEASE_NULL(attributes);
    CFRELEASE_NULL(uuid);
}

- (void)testSaveLoadHoldingCredential {
    HeimCredGlobalCTX.isMultiUser = NO;
    HeimCredGlobalCTX.useUidMatching = NO;
    [GSSCredTestUtil freePeer:self.peer];
    self.peer = [GSSCredTestUtil createPeer:@"com.apple.fake" identifier:0];

    CFUUIDRef uuid = NULL;
    BOOL worked = [GSSCredTestUtil createCredentialAndCache:self.peer name:@"test@EXAMPLE.COM" returningCacheUuid:&uuid];

    XCTAssertTrue(worked, "Credential should be created successfully");

    CFDictionaryRef attributes;
    worked = [GSSCredTestUtil fetchCredential:self.peer uuid:uuid returningDictionary:&attributes];
    XCTAssertTrue(worked, "Cache should be fetched successfully using it's uuid");

    int64_t error = [GSSCredTestUtil hold:self.peer uuid:uuid];
    XCTAssertEqual(error, 0, "hold should not error");

    CFRELEASE_NULL(attributes);
    worked = [GSSCredTestUtil fetchCredential:self.peer uuid:uuid returningDictionary:&attributes];
    XCTAssertTrue(worked, "Cache should still exist after hold");

    [GSSCredTestUtil freePeer:self.peer];

    CFRELEASE_NULL(HeimCredCTX.sessions);
    HeimCredCTX.sessions = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

    readCredCache();

    self.peer = [GSSCredTestUtil createPeer:@"com.apple.fake" identifier:0];

    CFRELEASE_NULL(attributes);
    worked = [GSSCredTestUtil fetchCredential:self.peer uuid:uuid returningDictionary:&attributes];
    XCTAssertTrue(worked, "Cache should still exist after hold");

    NSNumber *count = CFDictionaryGetValue(attributes, kHEIMAttrRetainStatus);

    XCTAssertEqual([count intValue], 2, "The retain count should match after save/load");

    CFRELEASE_NULL(uuid);
    CFRELEASE_NULL(attributes);
}

- (void)testUpdatingRetainStatusShouldFail {

    HeimCredGlobalCTX.isMultiUser = NO;
    HeimCredGlobalCTX.useUidMatching = YES;
    _currentUid = 501;
    [GSSCredTestUtil freePeer:self.peer];
    self.peer = [GSSCredTestUtil createPeer:@"com.apple.fake" identifier:0];

    CFUUIDRef uuid = NULL;
    BOOL worked = [GSSCredTestUtil createCredentialAndCache:self.peer name:@"test@EXAMPLE.COM" returningCacheUuid:&uuid];
    XCTAssertTrue(worked, "Credential was created successfully");

    NSDictionary *attributes = @{(id)kHEIMAttrRetainStatus:@10};
    uint64_t result = [GSSCredTestUtil setAttributes:self.peer uuid:uuid attributes:(__bridge CFDictionaryRef)(attributes) returningDictionary:NULL];
    XCTAssertEqual(result, kHeimCredErrorUpdateNotAllowed, "Updating the retain status should not be allowed.");

    CFRELEASE_NULL(uuid);
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

