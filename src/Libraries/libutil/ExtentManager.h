/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
//	ExtentManager.h
//
#ifndef EXTENTMANAGER_H
#define EXTENTMANAGER_H

#include <list>
#include <vector>
#include <algorithm>
#include <sys/types.h>
#include <sys/errno.h>
#include <cstdio>
#include <cassert>
using namespace std;

struct ExtentInfo {
	off_t blockAddr;
	off_t numBlocks;
};

inline bool BeforeExtent(const ExtentInfo &a, const ExtentInfo &b)
{
		return (a.blockAddr + a.numBlocks) < b.blockAddr;
}

typedef list<ExtentInfo>::iterator ListExtIt;

class ExtentManager {
public:
	ExtentManager() : blockSize(0), totalBytes(0), totalBlocks(0) {};
	~ExtentManager() {};

	void Init(uint32_t theBlockSize, uint32_t theNativeBlockSize, off_t theTotalBytes);

	void AddBlockRangeExtent(off_t blockAddr, off_t numBlocks);
	void AddByteRangeExtent(off_t byteAddr, off_t numBytes);
	void RemoveBlockRangeExtent(off_t blockAddr, off_t numBlocks);

	void DebugPrint();

protected:
	void MergeExtent(const ExtentInfo &a, const ExtentInfo &b, ExtentInfo *c);

public:
	size_t blockSize;
	size_t nativeBlockSize;
	off_t totalBytes;
	off_t totalBlocks;
	list<ExtentInfo> extentList;
};

#endif // #ifndef EXTENTMANAGER_H
