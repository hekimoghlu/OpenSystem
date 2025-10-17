/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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
#ifndef __IOKIT_IOMAPTYPES_H
#define __IOKIT_IOMAPTYPES_H

// IOConnectMapMemory memoryTypes
enum {
	kIODefaultMemoryType        = 0
};

enum {
	kIODefaultCache             = 0,
	kIOInhibitCache             = 1,
	kIOWriteThruCache           = 2,
	kIOCopybackCache            = 3,
	kIOWriteCombineCache        = 4,
	kIOCopybackInnerCache       = 5,
	kIOPostedWrite              = 6,
	kIORealTimeCache            = 7,
	kIOPostedReordered          = 8,
	kIOPostedCombinedReordered  = 9,
};

// IOMemory mapping options
enum {
	kIOMapAnywhere                = 0x00000001,

	kIOMapCacheMask               = 0x00000f00,
	kIOMapCacheShift              = 8,
	kIOMapDefaultCache            = kIODefaultCache            << kIOMapCacheShift,
	kIOMapInhibitCache            = kIOInhibitCache            << kIOMapCacheShift,
	kIOMapWriteThruCache          = kIOWriteThruCache          << kIOMapCacheShift,
	kIOMapCopybackCache           = kIOCopybackCache           << kIOMapCacheShift,
	kIOMapWriteCombineCache       = kIOWriteCombineCache       << kIOMapCacheShift,
	kIOMapCopybackInnerCache      = kIOCopybackInnerCache      << kIOMapCacheShift,
	kIOMapPostedWrite             = kIOPostedWrite             << kIOMapCacheShift,
	kIOMapRealTimeCache           = kIORealTimeCache           << kIOMapCacheShift,
	kIOMapPostedReordered         = kIOPostedReordered         << kIOMapCacheShift,
	kIOMapPostedCombinedReordered = kIOPostedCombinedReordered << kIOMapCacheShift,

	kIOMapUserOptionsMask         = 0x00000fff,

	kIOMapReadOnly                = 0x00001000,

	kIOMapStatic                  = 0x01000000,
	kIOMapReference               = 0x02000000,
	kIOMapUnique                  = 0x04000000,
#ifdef XNU_KERNEL_PRIVATE
	kIOMap64Bit                   = 0x08000000,
#endif
	kIOMapPrefault                = 0x10000000,
	kIOMapOverwrite               = 0x20000000,
	kIOMapGuardedMask             = 0xC0000000,
	kIOMapGuardedSmall            = 0x40000000,
	kIOMapGuardedLarge            = 0x80000000
};

#endif /* ! __IOKIT_IOMAPTYPES_H */
