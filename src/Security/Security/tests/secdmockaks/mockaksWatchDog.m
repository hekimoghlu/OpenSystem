/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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
//  mockaksWatchDog.m
//  Security
//

#import <XCTest/XCTest.h>
#import <OCMock/OCMock.h>

#import "mockaksxcbase.h"
#import "ipc/SecdWatchdog.h"

@interface mockaksWatchDog : mockaksxcbase
@property (assign) uint64_t diskusage;
@end

@implementation mockaksWatchDog

- (bool)mockedWatchdogrusage:(rusage_info_current *)rusage
{
    memset(rusage, 0, sizeof(*rusage));
    rusage->ri_diskio_byteswritten = self.diskusage;
    rusage->ri_logical_writes = self.diskusage;
    return true;
}


- (void)testWatchDogDiskWrite {

    id mock = OCMClassMock([SecdWatchdog class]);
    OCMStub([mock watchdogrusage:[OCMArg anyPointer]]).andCall(self, @selector(mockedWatchdogrusage:));
    OCMStub([mock triggerOSFaults]).andReturn(FALSE);

    SecdWatchdog *wd = [SecdWatchdog watchdog];

    self.diskusage = 0;
    XCTAssertFalse(wd.diskUsageHigh, "diskusage high should not be true");

    self.diskusage = 2 * 1000 * 1024 * 1024; // 2GiBi
    [wd runWatchdog];

    XCTAssertTrue(wd.diskUsageHigh, "diskusage high should be true");
}

@end
