/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 22, 2023.
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

//===--- IfConfigClauseRangeInfo.h - header for #if clauses -=====---------===//
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

#ifndef LANGUAGE_AST_IFCONFIGCLAUSERANGEINFO_H
#define LANGUAGE_AST_IFCONFIGCLAUSERANGEINFO_H

#include "language/Basic/SourceLoc.h"

namespace language {

class SourceManager;

/// Stores range information for a \c #if block in a SourceFile.
class IfConfigClauseRangeInfo final {
public:
  enum ClauseKind {
    // Active '#if', '#elseif', or '#else' clause.
    ActiveClause,
    // Inactive '#if', '#elseif', or '#else' clause.
    InactiveClause,
    // '#endif' directive.
    EndDirective,
  };

private:
  /// Source location of '#if', '#elseif', etc.
  SourceLoc DirectiveLoc;
  /// Character source location of body starts.
  SourceLoc BodyLoc;
  /// Location of the end of the body.
  SourceLoc EndLoc;

  ClauseKind Kind;

public:
  IfConfigClauseRangeInfo(SourceLoc DirectiveLoc, SourceLoc BodyLoc,
                          SourceLoc EndLoc, ClauseKind Kind)
      : DirectiveLoc(DirectiveLoc), BodyLoc(BodyLoc), EndLoc(EndLoc),
        Kind(Kind) {
    assert(DirectiveLoc.isValid() && BodyLoc.isValid() && EndLoc.isValid());
  }

  SourceLoc getStartLoc() const { return DirectiveLoc; }
  CharSourceRange getDirectiveRange(const SourceManager &SM) const;
  CharSourceRange getWholeRange(const SourceManager &SM) const;
  CharSourceRange getBodyRange(const SourceManager &SM) const;

  ClauseKind getKind() const { return Kind; }
};

}

#endif /* LANGUAGE_AST_IFCONFIGCLAUSERANGEINFO_H */
