/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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

//===--- RuleBuilder.h - Lowering desugared requirements to rules ---------===//
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

#ifndef LANGUAGE_RULEBUILDER_H
#define LANGUAGE_RULEBUILDER_H

#include "language/AST/ASTContext.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/DenseSet.h"
#include "toolchain/ADT/SmallVector.h"
#include <vector>
#include "RewriteContext.h"
#include "Rule.h"
#include "Symbol.h"
#include "Term.h"

namespace toolchain {
  class raw_ostream;
}

namespace language {

class AssociatedTypeDecl;
class ProtocolDecl;
class ProtocolTypeAlias;
class Requirement;

namespace rewriting {

/// A utility class for building rewrite rules from the top-level requirements
/// of a generic signature.
///
/// This also collects requirements from the transitive closure of all protocols
/// appearing on the right hand side of conformance requirements.
struct RuleBuilder {
  RewriteContext &Context;

  /// The transitive closure of all protocols appearing on the right hand
  /// side of conformance requirements.
  toolchain::DenseSet<const ProtocolDecl *> &ReferencedProtocols;

  /// A subset of the above in insertion order, consisting of the protocols
  /// whose rules we are going to import.
  ///
  /// If this is a rewrite system built from a generic signature, this vector
  /// contains all elements in the above set.
  ///
  /// If this is a rewrite system built from a strongly connected component
  /// of the protocol, this vector contains all elements in the above set
  /// except for the protocols belonging to the component representing the
  /// rewrite system itself; those protocols are added directly instead of
  /// being imported.
  std::vector<const ProtocolDecl *> ProtocolsToImport;

  /// The rules representing a complete rewrite system for the above vector,
  /// pulled in by collectRulesFromReferencedProtocols().
  std::vector<Rule> ImportedRules;

  /// New rules to add which will be marked 'permanent'. These are rules for
  /// introducing associated types, and relationships between layout,
  /// superclass and concrete type symbols. They are not eliminated by
  /// homotopy reduction, since they are always added when the rewrite system
  /// is built.
  std::vector<std::pair<MutableTerm, MutableTerm>> PermanentRules;

  /// New rules derived from requirements written by the user, which can be
  /// eliminated by homotopy reduction.
  std::vector<std::pair<MutableTerm, MutableTerm>> RequirementRules;

  /// Enables debugging output. Controlled by the -dump-requirement-machine
  /// frontend flag.
  unsigned Dump : 1;

  /// Used to ensure the initWith*() methods are only called once.
  unsigned Initialized : 1;

  RuleBuilder(RewriteContext &ctx,
              toolchain::DenseSet<const ProtocolDecl *> &referencedProtocols)
      : Context(ctx), ReferencedProtocols(referencedProtocols) {
    Dump = ctx.getASTContext().LangOpts.DumpRequirementMachine;
    Initialized = 0;
  }

  void initWithGenericSignature(ArrayRef<GenericTypeParamType *> genericParams,
                                ArrayRef<Requirement> requirements);
  void initWithWrittenRequirements(ArrayRef<GenericTypeParamType *> genericParams,
                                   ArrayRef<StructuralRequirement> requirements);
  void initWithProtocolSignatureRequirements(ArrayRef<const ProtocolDecl *> proto);
  void initWithProtocolWrittenRequirements(
      ArrayRef<const ProtocolDecl *> component,
      const toolchain::DenseMap<const ProtocolDecl *,
                           SmallVector<StructuralRequirement, 4>> protos);
  void initWithConditionalRequirements(ArrayRef<Requirement> requirements,
                                       ArrayRef<Term> substitutions);
  void addReferencedProtocol(const ProtocolDecl *proto);
  void collectRulesFromReferencedProtocols();
  void collectPackShapeRules(ArrayRef<GenericTypeParamType *> genericParams);

private:
  void addPermanentProtocolRules(const ProtocolDecl *proto);
  void addAssociatedType(const AssociatedTypeDecl *type,
                         const ProtocolDecl *proto);
  void
  addRequirement(const Requirement &req, const ProtocolDecl *proto,
                 std::optional<ArrayRef<Term>> substitutions = std::nullopt);
  void addRequirement(const StructuralRequirement &req,
                      const ProtocolDecl *proto);
  void addTypeAlias(const ProtocolTypeAlias &alias,
                    const ProtocolDecl *proto);
};

} // end namespace rewriting

} // end namespace language

#endif
