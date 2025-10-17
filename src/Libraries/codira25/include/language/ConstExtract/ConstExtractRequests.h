/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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

//===------- ConstExtractRequests.h - Extraction  Requests ------*- C++ -*-===//
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
//  This file defines const-extraction requests.
//
//===----------------------------------------------------------------------===//
#ifndef LANGUAGE_CONST_EXTRACT_REQUESTS_H
#define LANGUAGE_CONST_EXTRACT_REQUESTS_H

#include "language/AST/ASTTypeIDs.h"
#include "language/AST/ConstTypeInfo.h"
#include "language/AST/EvaluatorDependencies.h"
#include "language/AST/FileUnit.h"
#include "language/AST/Identifier.h"
#include "language/AST/NameLookup.h"
#include "language/AST/SimpleRequest.h"
#include "language/Basic/Statistic.h"
#include "toolchain/ADT/Hashing.h"
#include "toolchain/ADT/PointerUnion.h"
#include "toolchain/ADT/TinyPtrVector.h"

namespace language {

class Decl;
class DeclName;
class EnumDecl;

/// Retrieve information about compile-time-known values
class ConstantValueInfoRequest
    : public SimpleRequest<
          ConstantValueInfoRequest,
          ConstValueTypeInfo(
              NominalTypeDecl *,
              toolchain::PointerUnion<const SourceFile *, ModuleDecl *>),
          RequestFlags::Cached> {
public:
  using SimpleRequest::SimpleRequest;

private:
  friend SimpleRequest;

  // Evaluation.
  ConstValueTypeInfo
  evaluate(Evaluator &eval, NominalTypeDecl *nominal,
           toolchain::PointerUnion<const SourceFile *, ModuleDecl *> extractionScope)
      const;

public:
  // Caching
  bool isCached() const { return true; }
};

#define LANGUAGE_TYPEID_ZONE ConstExtract
#define LANGUAGE_TYPEID_HEADER "language/ConstExtract/ConstExtractTypeIDZone.def"
#include "language/Basic/DefineTypeIDZone.h"
#undef LANGUAGE_TYPEID_ZONE
#undef LANGUAGE_TYPEID_HEADER

// Set up reporting of evaluated requests.
template<typename Request>
void reportEvaluatedRequest(UnifiedStatsReporter &stats,
                            const Request &request);

#define LANGUAGE_REQUEST(Zone, RequestType, Sig, Caching, LocOptions)             \
  template <>                                                                  \
  inline void reportEvaluatedRequest(UnifiedStatsReporter &stats,              \
                                     const RequestType &request) {             \
    ++stats.getFrontendCounters().RequestType;                                 \
  }
#include "language/ConstExtract/ConstExtractTypeIDZone.def"
#undef LANGUAGE_REQUEST

} // end namespace language

#endif // LANGUAGE_CONST_EXTRACT_REQUESTS_H

