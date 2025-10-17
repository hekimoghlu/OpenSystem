/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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

//===--- CodiraInvocation.h - ------------------------------------*- C++ -*-===//
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

#ifndef TOOLCHAIN_SOURCEKIT_LIB_LANGUAGELANG_LANGUAGEINVOCATION_H
#define TOOLCHAIN_SOURCEKIT_LIB_LANGUAGELANG_LANGUAGEINVOCATION_H

#include "language/Basic/ThreadSafeRefCounted.h"
#include <string>
#include <vector>

namespace language {
  class CompilerInvocation;
}

namespace SourceKit {
  class CodiraASTManager;

/// Encompasses an invocation for getting an AST. This is used to control AST
/// sharing among different requests.
class CodiraInvocation : public toolchain::ThreadSafeRefCountedBase<CodiraInvocation> {
public:
  ~CodiraInvocation();

  struct Implementation;
  Implementation &Impl;

  ArrayRef<std::string> getArgs() const;
  void applyTo(language::CompilerInvocation &CompInvok) const;
  void raw(std::vector<std::string> &Args, std::string &PrimaryFile) const;

private:
  CodiraInvocation(Implementation &Impl) : Impl(Impl) { }
  friend class CodiraASTManager;
};

} // namespace SourceKit

#endif
