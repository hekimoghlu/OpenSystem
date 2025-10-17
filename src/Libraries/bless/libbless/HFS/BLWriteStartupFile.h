/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
/*
 *  writeStartupFile.h
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Mon Jun 25 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLWriteStartupFile.h,v 1.8 2006/02/20 22:49:55 ssen Exp $
 *
 */

// Defines
#ifndef BlockMoveData
#define BlockMoveData(s,d,l)    memcpy(d,s,l)
#endif

// Typedefs
typedef struct {
  /* File header */
  UInt16 magic;
#define kFileMagic                      0x1DF
  UInt16 nSections;
  UInt32 timeAndDate;
  UInt32 symPtr;
  UInt32 nSyms;
  UInt16 optHeaderSize;
  UInt16 flags;
} XFileHeader;

typedef struct {
  /* Optional header */
  UInt16 magic;
#define kOptHeaderMagic         0x10B
  UInt16 version;
  UInt32 textSize;
  UInt32 dataSize;
  UInt32 BSSSize;
  UInt32 entryPoint;
  UInt32 textStart;
  UInt32 dataStart;
  UInt32 toc;
  UInt16 snEntry;
  UInt16 snText;
  UInt16 snData;
  UInt16 snTOC;
  UInt16 snLoader;
  UInt16 snBSS;
  UInt8 filler[28];
} XOptHeader;

typedef struct {
  char name[8];
  UInt32 pAddr;
  UInt32 vAddr;
  UInt32 size;
  UInt32 sectionFileOffset;
  UInt32 relocationsFileOffset;
  UInt32 lineNumbersFileOffset;
  UInt16 nRelocations;
  UInt16 nLineNumbers;
  UInt32 flags;
} XSection;

enum SectionNumbers {
  kTextSN = 1,
  kDataSN,
  kBSSSN
};

const char kTextName[] = ".text";
const char kDataName[] = ".data";
const char kBSSName[] = ".bss";

/* @MAC_OS_X_END@ */
