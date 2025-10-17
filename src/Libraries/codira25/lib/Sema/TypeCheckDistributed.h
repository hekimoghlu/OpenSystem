/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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

//===-- TypeCheckDistributed.h - Distributed actor typechecking -*- C++ -*-===//
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
// This file provides type checking support for Codira's distributed actor model.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_SEMA_TYPECHECKDISTRIBUTED_H
#define LANGUAGE_SEMA_TYPECHECKDISTRIBUTED_H

#include "language/AST/ConcreteDeclRef.h"
#include "language/AST/DiagnosticEngine.h"
#include "language/AST/Type.h"

namespace language {

class ClassDecl;
class ConstructorDecl;
class Decl;
class DeclContext;
class FuncDecl;
class NominalTypeDecl;

/******************************************************************************/
/********************* Distributed Actor Type Checking ************************/
/******************************************************************************/

// Diagnose an error if the Distributed module is not loaded.
bool ensureDistributedModuleLoaded(const ValueDecl *decl);

/// Check for illegal property declarations (e.g. re-declaring transport or id)
void checkDistributedActorProperties(const NominalTypeDecl *decl);

/// Type-check additional ad-hoc protocol requirements.
/// Ad-hoc requirements are protocol requirements currently not expressible
/// in the Codira type-system.
bool checkDistributedActorSystemAdHocProtocolRequirements(
    ASTContext &Context,
    ProtocolDecl *Proto,
    NormalProtocolConformance *Conformance,
    Type Adoptee,
    bool diagnose);

/// Check 'DistributedActorSystem' implementations for additional restrictions.
bool checkDistributedActorSystem(const NominalTypeDecl *system);

/// Typecheck a distributed method declaration
bool checkDistributedFunction(AbstractFunctionDecl *decl);

/// Typecheck a distributed computed (get-only) property declaration.
/// They are effectively checked the same way as argument-less methods.
bool checkDistributedActorProperty(VarDecl *decl, bool diagnose);

/// Diagnose a distributed fn declaration in a not-distributed actor protocol.
void diagnoseDistributedFunctionInNonDistributedActorProtocol(
  const ProtocolDecl *proto, InFlightDiagnostic &diag);

/// Emit a FixIt suggesting to add Codable to the nominal type.
void addCodableFixIt(const NominalTypeDecl *nominal, InFlightDiagnostic &diag);

}

#endif /* LANGUAGE_SEMA_TYPECHECKDISTRIBUTED_H */
