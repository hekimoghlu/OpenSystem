/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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

//===--- SILOptimizerRequests.cpp - Requests for SIL Optimization  --------===//
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

#include "language/AST/SILOptimizerRequests.h"
#include "language/AST/Evaluator.h"
#include "language/SILOptimizer/PassManager/PassPipeline.h"
#include "language/Subsystems.h"
#include "toolchain/ADT/Hashing.h"

using namespace language;

namespace language {
// Implement the SILOptimizer type zone (zone 13).
#define LANGUAGE_TYPEID_ZONE SILOptimizer
#define LANGUAGE_TYPEID_HEADER "language/AST/SILOptimizerTypeIDZone.def"
#include "language/Basic/ImplementTypeIDZone.h"
#undef LANGUAGE_TYPEID_ZONE
#undef LANGUAGE_TYPEID_HEADER
} // end namespace language

//----------------------------------------------------------------------------//
// ExecuteSILPipelineRequest computation.
//----------------------------------------------------------------------------//

bool SILPipelineExecutionDescriptor::
operator==(const SILPipelineExecutionDescriptor &other) const {
  return SM == other.SM && Plan == other.Plan &&
         IsMandatory == other.IsMandatory && IRMod == other.IRMod;
}

toolchain::hash_code language::hash_value(const SILPipelineExecutionDescriptor &desc) {
  return toolchain::hash_combine(desc.SM, desc.Plan, desc.IsMandatory, desc.IRMod);
}

void language::simple_display(toolchain::raw_ostream &out,
                           const SILPipelineExecutionDescriptor &desc) {
  out << "Run pipelines { ";
  interleave(
      desc.Plan.getPipelines(),
      [&](SILPassPipeline stage) { out << stage.Name; },
      [&]() { out << ", "; });
  out << " } on ";
  simple_display(out, desc.SM);
}

SourceLoc
language::extractNearestSourceLoc(const SILPipelineExecutionDescriptor &desc) {
  return extractNearestSourceLoc(desc.SM);
}

//----------------------------------------------------------------------------//
// LoweredSILRequest computation.
//----------------------------------------------------------------------------//

std::unique_ptr<SILModule>
LoweredSILRequest::evaluate(Evaluator &evaluator,
                            ASTLoweringDescriptor desc) const {
  auto silMod = evaluateOrFatal(evaluator, ASTLoweringRequest{desc});
  silMod->installSILRemarkStreamer();
  silMod->setSerializeSILAction([]() {});

  runSILDiagnosticPasses(*silMod);

  {
    FrontendStatsTracer tracer(silMod->getASTContext().Stats,
                               "SIL verification, pre-optimization");
    silMod->verify();
  }

  runSILOptimizationPasses(*silMod);

  {
    FrontendStatsTracer tracer(silMod->getASTContext().Stats,
                               "SIL verification, post-optimization");
    silMod->verify();
  }

  runSILLoweringPasses(*silMod);
  return silMod;
}

// Define request evaluation functions for each of the SILGen requests.
static AbstractRequestFunction *silOptimizerRequestFunctions[] = {
#define LANGUAGE_REQUEST(Zone, Name, Sig, Caching, LocOptions)                    \
  reinterpret_cast<AbstractRequestFunction *>(&Name::evaluateRequest),
#include "language/AST/SILOptimizerTypeIDZone.def"
#undef LANGUAGE_REQUEST
};

void language::registerSILOptimizerRequestFunctions(Evaluator &evaluator) {
  evaluator.registerRequestFunctions(Zone::SILOptimizer,
                                     silOptimizerRequestFunctions);
}
