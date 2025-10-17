/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
//  hfs_xctests_utilities.h
//  hfs
//
//  Created by Adam Hijaze on 25/08/2022.
//

#ifndef hfs_xctests_utilities_h
#define hfs_xctests_utilities_h

#import <FSKitTesting/FSKitTesting.h>
#import <UVFSPluginTesting/UVFSPluginTests.h>
#import <UVFSPluginTesting/UVFSPluginFilesystem.h>
#import <UVFSPluginTesting/UVFSPluginPerformanceTests.h>
#include "lf_hfs_fsops_handler.h"

@interface HFSFactory : FSPluginInterfacesFactory
@end


@interface HFSSetupDelegate : FSPluginSetupDelegate

@property  TestVolumeUtils* volumeUtils;

@end

@interface HFSPluginSetupDelegate : HFSSetupDelegate
@end

@interface HFSModuleSetupDelegate : HFSSetupDelegate
@end

@interface HFSPluginFilesystem : UVFSPluginFilesystem
@end

#endif /* hfs_xctests_utilities_h */
