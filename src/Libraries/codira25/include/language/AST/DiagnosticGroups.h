/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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

//===--- DiagnosticGroups.h - Diagnostic Groups -----------------*- C++ -*-===//
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
//  This file defines the diagnostic groups enumaration, group graph
//  and auxilary functions.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_DIAGNOSTICGROUPS_H
#define LANGUAGE_DIAGNOSTICGROUPS_H

#include "toolchain/ADT/ArrayRef.h"
#include <array>
#include <string_view>
#include <unordered_map>

namespace language {
enum class DiagID : uint32_t;

enum class DiagGroupID : uint16_t {
#define GROUP(Name, Version) Name,
#include "language/AST/DiagnosticGroups.def"
};

constexpr const auto DiagGroupsCount = [] {
  size_t count = 0;
#define GROUP(Name, Version) ++count;
#include "DiagnosticGroups.def"
  return count;
}();

struct DiagGroupInfo {
  DiagGroupID id;
  std::string_view name;
  std::string_view documentationFile;
  toolchain::ArrayRef<DiagGroupID> supergroups;
  toolchain::ArrayRef<DiagGroupID> subgroups;
  toolchain::ArrayRef<DiagID> diagnostics;

  void traverseDepthFirst(
      toolchain::function_ref<void(const DiagGroupInfo &)> fn) const;
};

extern const std::array<DiagGroupInfo, DiagGroupsCount> diagnosticGroupsInfo;
const DiagGroupInfo &getDiagGroupInfoByID(DiagGroupID id);
std::optional<DiagGroupID> getDiagGroupIDByName(std::string_view name);

} // end namespace language

#endif /* LANGUAGE_DIAGNOSTICGROUPS_H */
