/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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

#import "FSPrivate.h"

#include <sys/mount.h>

@interface libfs_tests : XCTestCase

@end

@implementation libfs_tests

- (void)setUp {

    [super setUp];

    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.

    [super tearDown];
}

- (void)testTypeAndLocation_0 {

    /*
     * Test the basic "lifs" format for block device file systems.
     */

    struct statfs sfs = {
        .f_fstypename = "lifs",
        .f_fssubtype = 2,
        .f_mntfromname = "msdos://disk3s2/MYVOL",
    };

    char location[MNAMELEN];
    char typename[MFSTYPENAMELEN];
    uint32_t subtype;

    XCTAssert(_FSGetTypeInfoFromStatfs(&sfs, NULL, 0, NULL) == 0);
    XCTAssert(_FSGetTypeInfoFromStatfs(&sfs, typename, sizeof(typename), &subtype) == 0);
    XCTAssert(strcmp(typename, "msdos") == 0);
    XCTAssert(subtype == 2);

    XCTAssert(_FSGetLocationFromStatfs(&sfs, NULL, 0) == 0);
    XCTAssert(_FSGetLocationFromStatfs(&sfs, location, sizeof(location)) == 0);
    XCTAssert(strcmp(location, "disk3s2") == 0);
}

- (void)testTypeAndLocation_1 {

    /*
     * Test the basic FSKit format for block device file systems.
     */

    struct statfs sfs = {
        .f_fstypename = "fskit",
        .f_fssubtype = 2,
        .f_mntfromname = "msdos://disk3s2/MYVOL",
    };

    char location[MNAMELEN];
    char typename[MFSTYPENAMELEN];
    uint32_t subtype;

    XCTAssert(_FSGetTypeInfoFromStatfs(&sfs, typename, sizeof(typename), &subtype) == 0);
    XCTAssert(strcmp(typename, "msdos") == 0);
    XCTAssert(subtype == 2);

    XCTAssert(_FSGetLocationFromStatfs(&sfs, location, sizeof(location)) == 0);
    XCTAssert(strcmp(location, "disk3s2") == 0);
}

- (void)testTypeAndLocation_2 {

    /*
     * Test the basic KEXT format for block device file systems.
     */

    struct statfs sfs = {
        .f_fstypename = "hfs",
        .f_fssubtype = 0,
        .f_mntfromname = "/dev/disk5s1",
    };

    char location[MNAMELEN];
    char typename[MFSTYPENAMELEN];
    uint32_t subtype;

    XCTAssert(_FSGetTypeInfoFromStatfs(&sfs, typename, sizeof(typename), &subtype) == 0);
    XCTAssert(strcmp(typename, "hfs") == 0);
    XCTAssert(subtype == 0);

    XCTAssert(_FSGetLocationFromStatfs(&sfs, location, sizeof(location)) == 0);
    XCTAssert(strcmp(location, "disk5s1") == 0);
}

- (void)testTypeAndLocation_3 {

    /*
     * LiveFS / FSKit network file system.
     */

    struct statfs sfs = {
        .f_fstypename = "fskit",
        .f_mntfromname = "smb://user@server.com/SomeVolume",
    };

    char location[MNAMELEN];
    char typename[MFSTYPENAMELEN];
    uint32_t subtype;

    XCTAssert(_FSGetTypeInfoFromStatfs(&sfs, typename, sizeof(typename), &subtype) == 0);
    XCTAssert(strcmp(typename, "smb") == 0);
    XCTAssert(subtype == 0);

    XCTAssert(_FSGetLocationFromStatfs(&sfs, location, sizeof(location)) == 0);
    XCTAssert(strcmp(location, "user@server.com") == 0);
}

- (void)testTypeAndLocation_4 {

    /*
     * NFS!
     */

    struct statfs sfs = {
        .f_fstypename = "nfs",
        .f_mntfromname = "something.apple.com:/path/to/stuff",
    };

    char location[MNAMELEN];
    char typename[MFSTYPENAMELEN];
    uint32_t subtype;

    XCTAssert(_FSGetTypeInfoFromStatfs(&sfs, typename, sizeof(typename), &subtype) == 0);
    XCTAssert(strcmp(typename, "nfs") == 0);
    XCTAssert(subtype == 0);

    XCTAssert(_FSGetLocationFromStatfs(&sfs, location, sizeof(location)) == 0);
    XCTAssert(strcmp(location, "something.apple.com:/path/to/stuff") == 0);
}

- (void)testTypeAndLocation_5 {

    /*
     * Wacky case of "/" appearing in the "volume" position in statfs::f_mntfromname.
     */

    struct statfs sfs = {
        .f_fstypename = "fskit",
        .f_mntfromname = "apfs://disk4s2/my/Volume",
    };

    char location[MNAMELEN];
    char typename[MFSTYPENAMELEN];
    uint32_t subtype;

    XCTAssert(_FSGetTypeInfoFromStatfs(&sfs, typename, sizeof(typename), &subtype) == 0);
    XCTAssert(strcmp(typename, "apfs") == 0);
    XCTAssert(subtype == 0);

    XCTAssert(_FSGetLocationFromStatfs(&sfs, location, sizeof(location)) == 0);
    XCTAssert(strcmp(location, "disk4s2") == 0);
}

- (void)testTypeReusingBuffer {

    /*
     * Test using the statfs structure for the result storage.
     */

    struct statfs sfs = {
        .f_fstypename = "fskit",
        .f_fssubtype = 2,
        .f_mntfromname = "msdos://disk3s2/MYVOL",
    };

    XCTAssert(_FSGetTypeInfoFromStatfs(&sfs, sfs.f_fstypename,
                                       sizeof(sfs.f_fstypename),
                                       &sfs.f_fssubtype) == 0);
    XCTAssert(strcmp(sfs.f_fstypename, "msdos") == 0);
    XCTAssert(sfs.f_fssubtype == 2);
}

- (void)testLocationReusingBuffer {

    /*
     * Test using the statfs structure for the result storage.
     */

    struct statfs sfs = {
        .f_fstypename = "fskit",
        .f_fssubtype = 2,
        .f_mntfromname = "msdos://disk3s2/MYVOL",
    };

    XCTAssert(_FSGetLocationFromStatfs(&sfs, sfs.f_mntfromname,
                                       sizeof(sfs.f_mntfromname)) == 0);
    XCTAssert(strcmp(sfs.f_mntfromname, "disk3s2") == 0);
}

@end
