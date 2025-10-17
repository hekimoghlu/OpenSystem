/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 1, 2025.
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

//===--- Watchdog.h - Watchdog timer ----------------------------*- C++ -*-===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//
//
//  This file declares the toolchain::sys::Watchdog class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_WATCHDOG_H
#define LLVM_SUPPORT_WATCHDOG_H

#include <IndexStoreDB_LLVMSupport/toolchain_Support_Compiler.h>

namespace toolchain {
  namespace sys {

    /// This class provides an abstraction for a timeout around an operation
    /// that must complete in a given amount of time. Failure to complete before
    /// the timeout is an unrecoverable situation and no mechanisms to attempt
    /// to handle it are provided.
    class Watchdog {
    public:
      Watchdog(unsigned int seconds);
      ~Watchdog();
    private:
      // Noncopyable.
      Watchdog(const Watchdog &other) = delete;
      Watchdog &operator=(const Watchdog &other) = delete;
    };
  }
}

#endif
