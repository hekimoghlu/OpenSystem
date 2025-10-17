/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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

//===--- Statistic.h - ------------------------------------------*- C++ -*-===//
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

#ifndef TOOLCHAIN_SOURCEKIT_SUPPORT_STATISTIC_H
#define TOOLCHAIN_SOURCEKIT_SUPPORT_STATISTIC_H

#include "SourceKit/Support/UIdent.h"
#include <atomic>
#include <string>

namespace SourceKit {

struct Statistic {
  const UIdent name;
  const std::string description;
  std::atomic<int64_t> value = {0};

  Statistic(UIdent name, std::string description)
      : name(name), description(std::move(description)) {}

  int64_t operator++() {
    return 1 + value.fetch_add(1, std::memory_order_relaxed);
  }
  int64_t operator--() {
    return value.fetch_sub(1, std::memory_order_relaxed) - 1;
  }

  void updateMax(int64_t newValue) {
    int64_t prev = value.load(std::memory_order_relaxed);
    // Note: compare_exchange_weak updates 'prev' if it fails.
    while (newValue > prev && !value.compare_exchange_weak(
                                  prev, newValue, std::memory_order_relaxed)) {
    }
  }
};

} // namespace SourceKit

#endif // TOOLCHAIN_SOURCEKIT_SUPPORT_STATISTIC_H
