/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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
//  authdroottests.m
//
//

#import "authdtestsCommon.h"

@interface AuthorizationRootTests : XCTestCase
@end

@implementation AuthorizationRootTests

- (void)testAuthorizationCreateWithAuditToken {
    AuthorizationRef authRef = NULL;
    audit_token_t emptyToken = { { 0 } };
    
    OSStatus stat = AuthorizationCreateWithAuditToken(emptyToken, kAuthorizationEmptyEnvironment, kAuthorizationFlagDefaults, &authRef);
    XCTAssert(stat == errAuthorizationSuccess, "AuthorizationCreateWithAuditToken authRef for root process failed %d", stat);
}

- (void)testDatabaseProtection {
    CFDictionaryRef outDict = NULL;
    OSStatus status = AuthorizationRightGet(SAMPLE_RIGHT, &outDict);
    XCTAssert(status == errAuthorizationSuccess, "AuthorizationRightGet failed to get existing right %d", status);
    
    AuthorizationRef authRef;
    status = AuthorizationCreate(NULL, kAuthorizationEmptyEnvironment, kAuthorizationFlagDefaults, &authRef);
    XCTAssert(status == errAuthorizationSuccess, "AuthorizationCreate failed %d", status);
    
    // add a new right
    status = AuthorizationRightSet(authRef, NEW_RIGHT, outDict, NULL, NULL, NULL);
    XCTAssert(status == errAuthorizationSuccess, "AuthorizationRightSet failed to add a new right %d", status);
    
    // modify an existing right
    status = AuthorizationRightSet(authRef, UNPROTECTED_RIGHT, outDict, NULL, NULL, NULL);
    XCTAssert(status == errAuthorizationSuccess, "AuthorizationRightSet failed to update an unprotected right %d", status);
    
    // modify an existing protected right
    status = AuthorizationRightSet(authRef, PROTECTED_RIGHT, outDict, NULL, NULL, NULL);
    XCTAssert(status == errAuthorizationDenied, "AuthorizationRightSet failed to denial update of a protected right");
    
    AuthorizationFree(authRef, kAuthorizationFlagDefaults);
}

- (void) testAuthdLeaks {
    char *cmd = NULL;
    int ret = 0;
    FILE *fpipe;
    char *command = "pgrep ^authd";
    char c = 0;
    char buffer[256];
    UInt32 index = 0;
    pid_t pid;

    if (0 == (fpipe = (FILE*)popen(command, "r"))) {
        XCTFail("Unable to run pgrep");
    }

    memset(buffer, 0, sizeof(buffer));
    while (fread(&c, sizeof c, 1, fpipe)) {
        buffer[index++] = c;
    }
    pclose(fpipe);

    pid = atoi(buffer);
    XCTAssert(pid, "Unable to get authd PID");
    
    fprintf(stdout, "authd PID is %d", pid);
    
    asprintf(&cmd, "leaks %d", pid);
    if (cmd) {
        ret = system(cmd);
        free(cmd);
    }
    XCTAssert(ret == 0, "Leaks in authd detected");
}

@end
