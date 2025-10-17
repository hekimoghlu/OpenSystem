/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#ifndef DirBlock_h
#define DirBlock_h

#import "DirItem.h"

@interface DirBlock : NSObject

@property uint64_t offsetInVolume; /* Offset within the file system */

-(instancetype)initInDir:(DirItem *)dirItem;

-(void)releaseBlock;

/** Reads the specified dir block number (which is relative to volume start) into the dir block. */
-(NSError *)readDirBlockNum:(uint64_t)dirBlockNumberInVolume;

/** Reads the specified dir block number (which is relative to dir start) into the dir block. */
-(NSError *)readRelativeDirBlockNum:(uint32_t)dirBlockIdxInDir;

/** Returns a pointer to the dir block's data, at the given offset. */
-(void *)getBytesAtOffset:(uint64_t)offsetInDirBlock;

-(NSError *)setBytes:(NSData *)data
            atOffset:(uint64_t)offsetInDirBlock;

/** Write the whole dir block to disk. */
-(NSError *)writeToDisk;

/** Write a part of the dir block to disk. */
-(NSError *)writeToDiskFromOffset:(uint64_t)offsetInDirBlock
						   length:(uint64_t)lengthToWrite;

@end




#endif /* DirBlock_h */
