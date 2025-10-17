/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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
//  UniveralHelperTests.m
//  CodesigningUnitTests
//
//  Copyright (c) 2023 Apple. All rights reserved.
//
#import <XCTest/XCTest.h>

#include <security_utilities/unix++.h>
#include <security_utilities/macho++.h>

// A helper for 0 cast to the appropriate type for comparisons while testing.
#define ZERO ((unsigned long)0)

// A FAT binary available in every operating system.
const char *testBinaryPath = "/usr/libexec/amfid";

@interface UniveralHelperTests : XCTestCase
@end

@implementation UniveralHelperTests

- (void)testUniversalObjectInitializer {
    Security::UnixPlusPlus::AutoFileDesc fd(testBinaryPath, O_RDONLY);
    Universal univ = Universal(fd);

    MachO *mo = univ.architecture();
    XCTAssertGreaterThan(mo->length(), ZERO);

#if TARGET_OS_OSX
    // On Macs this should be universal and any arch
    // will have a non-zero offset.
    XCTAssert(univ.isUniversal());
    XCTAssertFalse(univ.narrowed());
    XCTAssertGreaterThan(mo->offset(), ZERO);
#else
    // On other platforms, this macho will be thin and
    // the offset will be 0.
    XCTAssertFalse(univ.isUniversal());
    XCTAssertFalse(univ.narrowed());
    XCTAssertEqual(mo->offset(), ZERO);
#endif
}

- (void)testUniversalHelperMachOCreation {
#if !TARGET_OS_OSX
    XCTSkip("Universal helper tests only work on macOS");
#endif

    // Make a standard universal object against a FAT binary.
    Security::UnixPlusPlus::AutoFileDesc fd(testBinaryPath, O_RDONLY);
    Universal univ = Universal(fd);
    XCTAssert(univ.isUniversal());
    XCTAssertFalse(univ.narrowed());

    // The universal object will indicate there are two architectures.
    Universal::Architectures arches;
    univ.architectures(arches);
    XCTAssertEqual(arches.size(), (unsigned long)2);

    Architecture x86Arch = Architecture("x86_64");
    Architecture armArch = Architecture("arm64e");

    // Now check that pulling out a macho for either architecture has
    // appropriate lengths and offsets.
    MachO *mo = univ.architecture(armArch);
    XCTAssertGreaterThan(mo->length(), ZERO);
    XCTAssertGreaterThan(mo->offset(), ZERO);
    mo = univ.architecture(x86Arch);
    XCTAssertGreaterThan(mo->length(), ZERO);
    XCTAssertGreaterThan(mo->offset(), ZERO);

    // Now check that if we ask for a MachO for a specific
    // slice offset, we get an object that is fully initialized.
    mo = univ.architecture(mo->offset());
    XCTAssertGreaterThan(mo->length(), ZERO);
    XCTAssertGreaterThan(mo->offset(), ZERO);

    // Now make a universal object against a FAT binary,
    // but manually specify a single offset/length.
    univ = Universal(fd, mo->offset(), mo->length());
    XCTAssertFalse(univ.isUniversal());
    XCTAssert(univ.narrowed());

    // If we ask for a macho now, for the single matching
    // architecture, we should get an object with appropriate
    // length and offset.
    mo = univ.architecture(x86Arch);
    XCTAssertGreaterThan(mo->length(), ZERO);
    XCTAssertGreaterThan(mo->offset(), ZERO);

    // And asking for the other architecture should
    // throw an exception.
    try {
        univ.architecture(armArch);
        XCTFail("Invalid arch should have thrown an exception");
    } catch (Security::UnixError err) {
        XCTAssertEqual(err.error, ENOEXEC);
    }
}

@end
