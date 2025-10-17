/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 31, 2024.
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
//  lf_hfs_file_extent_mapping.h
//  hfs
//
//  Created by Yakov Ben Zaken on 22/03/2018.
//

#ifndef lf_hfs_file_extent_mapping_h
#define lf_hfs_file_extent_mapping_h

/*    File Extent Mapping routines*/
OSErr FlushExtentFile( ExtendedVCB *vcb );

int32_t CompareExtentKeysPlus( const HFSPlusExtentKey *searchKey, const HFSPlusExtentKey *trialKey );

OSErr SearchExtentFile( ExtendedVCB             *vcb,
                       const FCB               *fcb,
                       int64_t                 filePosition,
                       HFSPlusExtentKey        *foundExtentKey,
                       HFSPlusExtentRecord     foundExtentData,
                       u_int32_t               *foundExtentDataIndex,
                       u_int32_t               *extentBTreeHint,
                       u_int32_t               *endingFABNPlusOne );

OSErr TruncateFileC(    ExtendedVCB         *vcb,
                    FCB                 *fcb,
                    int64_t             peof,
                    int                 deleted,
                    int                 rsrc,
                    uint32_t            fileid,
                    Boolean             truncateToExtent );

OSErr ExtendFileC(  ExtendedVCB     *vcb,
                  FCB             *fcb,
                  int64_t         bytesToAdd,
                  u_int32_t       blockHint,
                  u_int32_t       flags,
                  int64_t         *actualBytesAdded );

OSErr MapFileBlockC(    ExtendedVCB     *vcb,
                    FCB             *fcb,
                    size_t          numberOfBytes,
                    off_t           offset,
                    daddr64_t       *startBlock,
                    size_t          *availableBytes );

OSErr HeadTruncateFile( ExtendedVCB     *vcb,
                       FCB             *fcb,
                       u_int32_t       headblks );

int AddFileExtent(  ExtendedVCB     *vcb,
                  FCB             *fcb,
                  u_int32_t       startBlock,
                  u_int32_t       blockCount );

Boolean NodesAreContiguous( ExtendedVCB     *vcb,
                           FCB             *fcb,
                           u_int32_t       nodeSize );

#endif /* lf_hfs_file_extent_mapping_h */
