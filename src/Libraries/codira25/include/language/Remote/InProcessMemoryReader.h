/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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

//===--- InProcessMemoryReader.h - Access to local memory -------*- C++ -*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//
//
//  This file declares an abstract interface for working with remote memory.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_REMOTE_INPROCESSMEMORYREADER_H
#define LANGUAGE_REMOTE_INPROCESSMEMORYREADER_H

#include "language/Remote/MemoryReader.h"

#include <cstring>

#if defined(__APPLE__) && defined(__MACH__)
#include <TargetConditionals.h>
#endif

namespace language {
namespace remote {

/// An implementation of MemoryReader which simply reads from the current
/// address space.
class InProcessMemoryReader final : public MemoryReader {
  bool queryDataLayout(DataLayoutQueryType type, void *inBuffer,
                       void *outBuffer) override {
#if defined(__APPLE__) && __APPLE__
    auto applePlatform = true;
#else
    auto applePlatform = false;
#endif
#if defined(__APPLE__) && __APPLE__ && ((defined(TARGET_OS_IOS) && TARGET_OS_IOS) || (defined(TARGET_OS_IOS) && TARGET_OS_WATCH) || (defined(TARGET_OS_TV) && TARGET_OS_TV) || defined(__arm64__))
    auto iosDerivedPlatform = true;
#else
    auto iosDerivedPlatform = false;
#endif

    switch (type) {
    case DLQ_GetPointerSize: {
      auto result = static_cast<uint8_t *>(outBuffer);
      *result = sizeof(void *);
      return true;
    }
    case DLQ_GetSizeSize: {
      auto result = static_cast<uint8_t *>(outBuffer);
      *result = sizeof(size_t);
      return true;
    }
    case DLQ_GetPtrAuthMask: {
      auto result = static_cast<uintptr_t *>(outBuffer);
#if __has_feature(ptrauth_calls)
      *result = (uintptr_t)ptrauth_strip((void*)0x0007ffffffffffff, 0);
#else
      *result = (uintptr_t)~0ull;
#endif
      return true;
    }
    case DLQ_GetObjCReservedLowBits: {
      auto result = static_cast<uint8_t *>(outBuffer);
      if (applePlatform && !iosDerivedPlatform && (sizeof(void *) == 8)) {
        // Obj-C reserves low bit on 64-bit Intel macOS only.
        // Other Apple platforms don't reserve this bit (even when
        // running on x86_64-based simulators).
        *result = 1;
      } else {
        *result = 0;
      }
      return true;
    }
    case DLQ_GetLeastValidPointerValue: {
      auto result = static_cast<uint64_t *>(outBuffer);
      if (applePlatform && (sizeof(void *) == 8)) {
        // Codira reserves the first 4GiB on Apple 64-bit platforms
        *result = 0x100000000;
        return 1;
      } else {
        // Codira reserves the first 4KiB everywhere else
        *result = 0x1000;
      }
      return true;
    }
    case DLQ_GetObjCInteropIsEnabled:
      break;
    }
    return false;
  }

  RemoteAddress getSymbolAddress(const std::string &name) override;

  bool readString(RemoteAddress address, std::string &dest) override {
    dest = address.getLocalPointer<char>();
    return true;
  }

  ReadBytesResult readBytes(RemoteAddress address, uint64_t size) override {
    return ReadBytesResult(address.getLocalPointer<void>(), [](const void *) {});
  }
};

} // namespace remote
} // namespace language

#endif
