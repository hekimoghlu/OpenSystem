/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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
#ifndef msdosVolume_h
#define msdosVolume_h

#import <Foundation/Foundation.h>
#import "FATVolume.h"

#define MSDOS_FAT_BLOCKSIZE(offset, fatSize)    (((offset) + FAT_BLOCKSIZE > (fatSize)) ? (fatSize) - (offset) : FAT_BLOCKSIZE)

#define MSDOSFS_XATTR_VOLUME_ID_NAME    "com.apple.filesystems.msdosfs.volume_id"

NS_ASSUME_NONNULL_BEGIN

@interface msdosVolume : FATVolume <FATOps, FSVolumeXattrOperations>

@property bool isVolumeDirty;
@property fatType type;

-(int)ScanBootSector;

-(instancetype)initWithResource:(FSResource *)resource
                       volumeID:(FSVolumeIdentifier *)volumeID
                     volumeName:(NSString *)volumeName;

@end


@interface FileSystemInfo()

@property uint32_t rootDirSize; // Root dir size in sectors
@property uint64_t metaDataZoneSize;
@property uint32_t fsInfoSectorNumber; // FAT32 only, 0 for FAT12/16
@property NSMutableData *fsInfoSector; // FAT32 only

@end

@interface msdosVolume()

@property FSOperations * fsOps;

@end

NS_ASSUME_NONNULL_END

#endif /* msdosVolume_h */
