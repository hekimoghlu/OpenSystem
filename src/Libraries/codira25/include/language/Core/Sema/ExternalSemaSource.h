/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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

//===--- ExternalSemaSource.h - External Sema Interface ---------*- C++ -*-===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ExternalSemaSource interface.
//
//===----------------------------------------------------------------------===//
#ifndef LANGUAGE_CORE_SEMA_EXTERNALSEMASOURCE_H
#define LANGUAGE_CORE_SEMA_EXTERNALSEMASOURCE_H

#include "language/Core/AST/ExternalASTSource.h"
#include "language/Core/AST/Type.h"
#include "language/Core/Sema/TypoCorrection.h"
#include "language/Core/Sema/Weak.h"
#include "toolchain/ADT/MapVector.h"
#include <utility>

namespace toolchain {
template <class T, unsigned n> class SmallSetVector;
}

namespace language::Core {

class CXXConstructorDecl;
class CXXRecordDecl;
class DeclaratorDecl;
class LookupResult;
class Scope;
class Sema;
class TypedefNameDecl;
class ValueDecl;
class VarDecl;
struct LateParsedTemplate;

/// A simple structure that captures a vtable use for the purposes of
/// the \c ExternalSemaSource.
struct ExternalVTableUse {
  CXXRecordDecl *Record;
  SourceLocation Location;
  bool DefinitionRequired;
};

/// An abstract interface that should be implemented by
/// external AST sources that also provide information for semantic
/// analysis.
class ExternalSemaSource : public ExternalASTSource {
  /// LLVM-style RTTI.
  static char ID;

public:
  ExternalSemaSource() = default;

  ~ExternalSemaSource() override;

  /// Initialize the semantic source with the Sema instance
  /// being used to perform semantic analysis on the abstract syntax
  /// tree.
  virtual void InitializeSema(Sema &S) {}

  /// Inform the semantic consumer that Sema is no longer available.
  virtual void ForgetSema() {}

  /// Load the contents of the global method pool for a given
  /// selector.
  virtual void ReadMethodPool(Selector Sel);

  /// Load the contents of the global method pool for a given
  /// selector if necessary.
  virtual void updateOutOfDateSelector(Selector Sel);

  /// Load the set of namespaces that are known to the external source,
  /// which will be used during typo correction.
  virtual void ReadKnownNamespaces(
                           SmallVectorImpl<NamespaceDecl *> &Namespaces);

  /// Load the set of used but not defined functions or variables with
  /// internal linkage, or used but not defined internal functions.
  virtual void
  ReadUndefinedButUsed(toolchain::MapVector<NamedDecl *, SourceLocation> &Undefined);

  virtual void ReadMismatchingDeleteExpressions(toolchain::MapVector<
      FieldDecl *, toolchain::SmallVector<std::pair<SourceLocation, bool>, 4>> &);

  /// Do last resort, unqualified lookup on a LookupResult that
  /// Sema cannot find.
  ///
  /// \param R a LookupResult that is being recovered.
  ///
  /// \param S the Scope of the identifier occurrence.
  ///
  /// \return true to tell Sema to recover using the LookupResult.
  virtual bool LookupUnqualified(LookupResult &R, Scope *S) { return false; }

  /// Read the set of tentative definitions known to the external Sema
  /// source.
  ///
  /// The external source should append its own tentative definitions to the
  /// given vector of tentative definitions. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadTentativeDefinitions(
                                  SmallVectorImpl<VarDecl *> &TentativeDefs) {}

  /// Read the set of unused file-scope declarations known to the
  /// external Sema source.
  ///
  /// The external source should append its own unused, filed-scope to the
  /// given vector of declarations. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadUnusedFileScopedDecls(
                 SmallVectorImpl<const DeclaratorDecl *> &Decls) {}

  /// Read the set of delegating constructors known to the
  /// external Sema source.
  ///
  /// The external source should append its own delegating constructors to the
  /// given vector of declarations. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadDelegatingConstructors(
                 SmallVectorImpl<CXXConstructorDecl *> &Decls) {}

  /// Read the set of ext_vector type declarations known to the
  /// external Sema source.
  ///
  /// The external source should append its own ext_vector type declarations to
  /// the given vector of declarations. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadExtVectorDecls(SmallVectorImpl<TypedefNameDecl *> &Decls) {}

  /// Read the set of potentially unused typedefs known to the source.
  ///
  /// The external source should append its own potentially unused local
  /// typedefs to the given vector of declarations. Note that this routine may
  /// be invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadUnusedLocalTypedefNameCandidates(
      toolchain::SmallSetVector<const TypedefNameDecl *, 4> &Decls) {}

  /// Read the set of referenced selectors known to the
  /// external Sema source.
  ///
  /// The external source should append its own referenced selectors to the
  /// given vector of selectors. Note that this routine
  /// may be invoked multiple times; the external source should take care not
  /// to introduce the same selectors repeatedly.
  virtual void ReadReferencedSelectors(
                 SmallVectorImpl<std::pair<Selector, SourceLocation> > &Sels) {}

  /// Read the set of weak, undeclared identifiers known to the
  /// external Sema source.
  ///
  /// The external source should append its own weak, undeclared identifiers to
  /// the given vector. Note that this routine may be invoked multiple times;
  /// the external source should take care not to introduce the same identifiers
  /// repeatedly.
  virtual void ReadWeakUndeclaredIdentifiers(
                 SmallVectorImpl<std::pair<IdentifierInfo *, WeakInfo> > &WI) {}

  /// Read the set of used vtables known to the external Sema source.
  ///
  /// The external source should append its own used vtables to the given
  /// vector. Note that this routine may be invoked multiple times; the external
  /// source should take care not to introduce the same vtables repeatedly.
  virtual void ReadUsedVTables(SmallVectorImpl<ExternalVTableUse> &VTables) {}

  /// Read the set of pending instantiations known to the external
  /// Sema source.
  ///
  /// The external source should append its own pending instantiations to the
  /// given vector. Note that this routine may be invoked multiple times; the
  /// external source should take care not to introduce the same instantiations
  /// repeatedly.
  virtual void ReadPendingInstantiations(
                 SmallVectorImpl<std::pair<ValueDecl *,
                                           SourceLocation> > &Pending) {}

  /// Read the set of late parsed template functions for this source.
  ///
  /// The external source should insert its own late parsed template functions
  /// into the map. Note that this routine may be invoked multiple times; the
  /// external source should take care not to introduce the same map entries
  /// repeatedly.
  virtual void ReadLateParsedTemplates(
      toolchain::MapVector<const FunctionDecl *, std::unique_ptr<LateParsedTemplate>>
          &LPTMap) {}

  /// Read the set of decls to be checked for deferred diags.
  ///
  /// The external source should append its own potentially emitted function
  /// and variable decls which may cause deferred diags. Note that this routine
  /// may be invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void
  ReadDeclsToCheckForDeferredDiags(toolchain::SmallSetVector<Decl *, 4> &Decls) {}

  /// \copydoc Sema::CorrectTypo
  /// \note LookupKind must correspond to a valid Sema::LookupNameKind
  ///
  /// ExternalSemaSource::CorrectTypo is always given the first chance to
  /// correct a typo (really, to offer suggestions to repair a failed lookup).
  /// It will even be called when SpellChecking is turned off or after a
  /// fatal error has already been detected.
  virtual TypoCorrection CorrectTypo(const DeclarationNameInfo &Typo,
                                     int LookupKind, Scope *S, CXXScopeSpec *SS,
                                     CorrectionCandidateCallback &CCC,
                                     DeclContext *MemberContext,
                                     bool EnteringContext,
                                     const ObjCObjectPointerType *OPT) {
    return TypoCorrection();
  }

  /// Produces a diagnostic note if the external source contains a
  /// complete definition for \p T.
  ///
  /// \param Loc the location at which a complete type was required but not
  /// provided
  ///
  /// \param T the \c QualType that should have been complete at \p Loc
  ///
  /// \return true if a diagnostic was produced, false otherwise.
  virtual bool MaybeDiagnoseMissingCompleteType(SourceLocation Loc,
                                                QualType T) {
    return false;
  }

  /// Notify the external source that a lambda was assigned a mangling number.
  /// This enables the external source to track the correspondence between
  /// lambdas and mangling numbers if necessary.
  virtual void AssignedLambdaNumbering(CXXRecordDecl *Lambda) {}

  /// LLVM-style RTTI.
  /// \{
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || ExternalASTSource::isA(ClassID);
  }
  static bool classof(const ExternalASTSource *S) { return S->isA(&ID); }
  /// \}
};

} // end namespace language::Core

#endif
