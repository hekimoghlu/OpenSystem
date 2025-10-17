/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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

//===--- LexerState.h - Lexer State -----------------------------*- C++ -*-===//
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
//  This file defines the LexerState object.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_LEXERSTATE_H
#define LANGUAGE_LEXERSTATE_H

#include "language/Basic/SourceLoc.h"
#include <optional>

namespace language {
class Lexer;

/// Lexer state can be saved/restored to/from objects of this class.
class LexerState {
public:
  LexerState() {}

  bool isValid() const { return Loc.isValid(); }

  LexerState advance(unsigned Offset) const {
    assert(isValid());
    return LexerState(Loc.getAdvancedLoc(Offset));
  }

private:
  explicit LexerState(SourceLoc Loc) : Loc(Loc) {}
  SourceLoc Loc;
  friend class Lexer;
};

} // end namespace language

#endif
