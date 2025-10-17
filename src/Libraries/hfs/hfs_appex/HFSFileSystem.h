/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
//  HFSFileSystem.h
//  hfs_xctests
//
//  Created by Adam Hijaze on 24/08/2022.
//

#import <Foundation/Foundation.h>
#import <FSKit/FSKit.h>
#import <FSKit/FSKit_private.h>

#include "lf_hfs_endian.h"
#include "lf_hfs_vfsutils.h"
#include "lf_hfs_volume_identifiers.h"

NS_ASSUME_NONNULL_BEGIN

#define kMaxLogicalBlockSize (16*1024)


@interface HFSFileSystem : FSUnaryFileSystem <FSUnaryFileSystemOperations, FSBlockDeviceOperations>

@end

NS_ASSUME_NONNULL_END
