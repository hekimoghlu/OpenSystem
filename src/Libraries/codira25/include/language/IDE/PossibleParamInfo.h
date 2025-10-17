/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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

//===--- PossibleParamInfo.h ----------------------------------------------===//
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

#ifndef LANGUAGE_IDE_POSSIBLEPARAMINFO_H
#define LANGUAGE_IDE_POSSIBLEPARAMINFO_H

#include "language/AST/Types.h"

namespace language {
namespace ide {

struct PossibleParamInfo {
  /// Expected parameter.
  ///
  /// 'nullptr' indicates that the code completion position is at out of
  /// expected argument position. E.g.
  ///   fn foo(x: Int) {}
  ///   foo(x: 1, <HERE>)
  const AnyFunctionType::Param *Param;
  bool IsRequired;

  PossibleParamInfo(const AnyFunctionType::Param *Param, bool IsRequired)
      : Param(Param), IsRequired(IsRequired) {
    assert((Param || !IsRequired) &&
           "nullptr with required flag is not allowed");
  };

  friend bool operator==(const PossibleParamInfo &lhs,
                         const PossibleParamInfo &rhs) {
    bool ParamsMatch;
    if (lhs.Param == nullptr && rhs.Param == nullptr) {
      ParamsMatch = true;
    } else if (lhs.Param == nullptr || rhs.Param == nullptr) {
      // One is nullptr but the other is not.
      ParamsMatch = false;
    } else {
      // Both are not nullptr.
      ParamsMatch = (*lhs.Param == *rhs.Param);
    }
    return ParamsMatch && (lhs.IsRequired == rhs.IsRequired);
  }
};

} // end namespace ide
} // end namespace language

#endif // LANGUAGE_IDE_POSSIBLEPARAMINFO_H
