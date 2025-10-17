/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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
#ifndef msdosItem_h
#define msdosItem_h

#import <Foundation/Foundation.h>
#import <FSKit/FSKit.h>
#import "ExtensionCommon.h"
#import "msdosVolume.h"
#import "direntry.h"
#import "FATItem.h"
#import "DirItem.h"

NS_ASSUME_NONNULL_BEGIN

@interface MsdosDirEntryData : DirEntryData

@property uint64_t dosDirEntryOffsetInDirBlock; /* offset in dir block of the dosdirentry data */
@property uint64_t dosDirEntryDirBlockNum; /* dir block number (in volume) which holds the dosdirentry data */

@end

@interface MsdosDirItem: DirItem

/*
 * Instead of iterate the directory for every shortname looking for the next
 * available generation number, use a single counter for the entire directory.
 * If we wrap around, iterate the directory next time.
 */
@property uint32_t maxShortNameIndex;

@end


@interface MsdosFileItem : FileItem

-(void)waitForWrites;

@end

NS_ASSUME_NONNULL_END
#endif /* msdosItem_h */
