/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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
//  hfs_xctests.m
//  hfs_xctests
//
//  Created by Kujan Lauz on 07/12/2022.
//

#import "hfs_xctests.h"
#import "hfs_xctests_utilities.h"


@implementation HFSUnitTests

-(void)testFileChangeModeUpdateChangeTime {
    UVFSFileNode fileNode = NULL;
    UVFSFileNode dirNode = NULL;
    const char* fileName = "testFile";
    const char* dirName = "testDir";
    UVFSFileAttributes attrsToSet;
    UVFSFileAttributes outAttrs;
    struct timespec tsAttrChange = {0};
    
    memset(&attrsToSet, 0, sizeof(UVFSFileAttributes));
    
    FSKIT_TEST_LOG("Creating dir and file");
    PRINT_IF_FAILED([self.fs createNewFolder:self.fs.rootNode dirNode:&dirNode name:dirName]);
    PRINT_IF_FAILED([self.fs createNewFile:self.fs.rootNode newNode:&fileNode name:fileName size:0]);
    
    FSKIT_TEST_LOG("Sleeping to make a time difference before changing mode");
    sleep(3);
    
    FSKIT_TEST_LOG("Changing dir and file mode");
    getTime(&tsAttrChange);
    attrsToSet.fa_validmask = UVFS_FA_VALID_MODE;
    attrsToSet.fa_mode = UVFS_FA_MODE_USR(UVFS_FA_MODE_RWX);
    PRINT_IF_FAILED([self.fs setAttr:fileNode attrs:&attrsToSet outAttrs:&outAttrs]);
    XCTAssertEqual(outAttrs.fa_mode, UVFS_FA_MODE_USR(UVFS_FA_MODE_RWX));
    PRINT_IF_FAILED([self.fs setAttr:dirNode attrs:&attrsToSet outAttrs:&outAttrs]);
    XCTAssertEqual(outAttrs.fa_mode, UVFS_FA_MODE_USR(UVFS_FA_MODE_RWX));
    
    FSKIT_TEST_LOG("Checking dir and file change time (fa_ctime)");
    PRINT_IF_FAILED([self.fs getAttr:dirNode outAttrs:&outAttrs]);
    XCTAssert(labs(outAttrs.fa_ctime.tv_sec - tsAttrChange.tv_sec) < 3, "The Dir access timestamp is different than expected. expected (%ld) -/+3 but got (%ld)", tsAttrChange.tv_sec, outAttrs.fa_ctime.tv_sec);
    PRINT_IF_FAILED([self.fs getAttr:fileNode outAttrs:&outAttrs]);
    XCTAssert(labs(outAttrs.fa_ctime.tv_sec - tsAttrChange.tv_sec) < 3, "The File access timestamp is different than expected. expected (%ld) -/+3 but got (%ld)", tsAttrChange.tv_sec, outAttrs.fa_ctime.tv_sec);
    
    FSKIT_TEST_LOG("Cleanup");
    PRINT_IF_FAILED([self.fs removeFile:self.fs.rootNode name:fileName victim:fileNode]);
    PRINT_IF_FAILED([FSTestOpsUtils closeFile:self.fs node:fileNode flags:0]);
    PRINT_IF_FAILED([self.fs removeFolder:self.fs.rootNode dirName:dirName victim:dirNode]);
}

@end

@implementation HFSModuleUnitTests

-(void)setUp {
    [super setDelegate:[[HFSModuleSetupDelegate alloc] init]];
    [super setUp];
}

@end

@implementation HFSPluginUnitTests

-(void)setUp {
    // If test name contains blockmap, skip it.
    if ([self.name rangeOfString:@"blockmap" options:NSCaseInsensitiveSearch].location != NSNotFound)
    {
        XCTSkip("Blockmap is not supported on HFS");
    }
    [super setDelegate:[[HFSPluginSetupDelegate alloc] init]];
    [super setUp];
}

@end


@implementation HFSPerformanceTests

-(void)setUp {
    [super setDelegate:[[HFSSetupDelegate alloc] init]];
    [super setUp];
}

@end
