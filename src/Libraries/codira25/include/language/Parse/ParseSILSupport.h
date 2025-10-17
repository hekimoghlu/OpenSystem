/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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

//===--- ParseSILSupport.h - Interface with ParseSIL ------------*- C++ -*-===//
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

#ifndef LANGUAGE_PARSER_PARSESILSUPPORT_H
#define LANGUAGE_PARSER_PARSESILSUPPORT_H

#include "toolchain/Support/PrettyStackTrace.h"

namespace language {
  class Parser;

  /// Interface between the Parse and ParseSIL libraries, to avoid circular
  /// dependencies.
  class SILParserStateBase {
    virtual void anchor();
  protected:
    SILParserStateBase() = default;
    virtual ~SILParserStateBase() = default;
  public:
    virtual bool parseDeclSIL(Parser &P) = 0;
    virtual bool parseDeclSILStage(Parser &P) = 0;
    virtual bool parseSILVTable(Parser &P) = 0;
    virtual bool parseSILMoveOnlyDeinit(Parser &P) = 0;
    virtual bool parseSILGlobal(Parser &P) = 0;
    virtual bool parseSILWitnessTable(Parser &P) = 0;
    virtual bool parseSILDefaultWitnessTable(Parser &P) = 0;
    virtual bool parseSILDefaultOverrideTable(Parser &P) = 0;
    virtual bool parseSILDifferentiabilityWitness(Parser &P) = 0;
    virtual bool parseSILCoverageMap(Parser &P) = 0;
    virtual bool parseSILProperty(Parser &P) = 0;
    virtual bool parseSILScope(Parser &P) = 0;
  };
} // end namespace language

#endif // LANGUAGE_PARSER_PARSESILSUPPORT_H

