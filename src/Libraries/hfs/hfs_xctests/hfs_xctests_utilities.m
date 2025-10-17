/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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
//  hfs_xctests_utilities.m
//  hfs_xctests
//
//  Created by Tomer Afek on 07/14/2022.
//

#import "hfs_xctests_utilities.h"
#import "hfsFileSystem.h"

@implementation HFSPluginFilesystem

// Intentionally empty

@end


@implementation HFSFactory

// TODO: Implement this when enabling all the tests.

@end

@implementation HFSSetupDelegate

-(void)deinit
{
    [self.volumeUtils clearTestVolume];
}

-(FSPluginInterfacesFactory *)getFactory:(id<FSTestOps>)fs
{
    return [[HFSFactory alloc] initWithFS:fs];
}

@end

@implementation HFSPluginSetupDelegate

-(instancetype)init
{
    self = [super init];
    if(self) {
        self.volumeUtils = [[TestVolumeUtils alloc] initTestVolume:@"hfs"
                                                              size:@"2G"
                                                         newfsPath:@"/sbin/newfs_hfs"];
    }
    return self;
}

-(id<FSTestOps>)getFileSystem
{
    return [[HFSPluginFilesystem alloc] initWithFSOps:&HFS_fsOps
                                           devicePath:self.volumeUtils.devicePath
                                           volumeName:self.volumeUtils.volumeName
                                            newfsPath:self.volumeUtils.newfsPath];
}

@end

@implementation HFSModuleSetupDelegate

-(instancetype)init
{
    self = [super init];
    if(self) {
        self.volumeUtils = [[TestVolumeUtils alloc] initTestVolume:@"hfs"
                                                              size:@"2G"
                                                         newfsPath:nil];
    }
    return self;
}

-(id<FSTestOps>)getFileSystem
{
    return [[FSTestModuleOps alloc] initWithFSModule:[[HFSFileSystem alloc] init]
                                          devicePath:self.volumeUtils.volumeName
                                          volumeName:self.volumeUtils.newfsPath];
}
@end

