/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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

//===--- ClangModuleLoader.h - Clang Module Loader Interface ----*- C++ -*-===//
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

#ifndef LANGUAGE_AST_CLANG_MODULE_LOADER_H
#define LANGUAGE_AST_CLANG_MODULE_LOADER_H

#include "language/AST/ModuleLoader.h"
#include "language/AST/SubstitutionMap.h"
#include "language/Basic/TaggedUnion.h"
#include "language/Core/AST/DeclTemplate.h"

namespace language::Core {
class ASTContext;
class CompilerInstance;
class Decl;
class Module;
class Preprocessor;
class Sema;
class TargetInfo;
class Type;
class SourceLocation;
} // namespace language::Core

namespace language {

class ClangInheritanceInfo;
class ClangNode;
class ConcreteDeclRef;
class Decl;
class FuncDecl;
class VarDecl;
class DeclContext;
class EffectiveClangContext;
class CodiraLookupTable;
class ValueDecl;
class VisibleDeclConsumer;

/// Represents the different namespaces for types in C.
///
/// A simplified version of language::Core::Sema::LookupKind.
enum class ClangTypeKind {
  Typedef,
  ObjCClass = Typedef,
  /// Structs, enums, and unions.
  Tag,
  ObjCProtocol,
};

/// A path for serializing a declaration.
class StableSerializationPath {
public:
  struct ExternalPath {
    enum ComponentKind {
      /// A named record type (but not a template specialization)
      Record,

      /// A named enum type
      Enum,

      /// A C++ namespace
      Namespace,

      /// A typedef
      Typedef,

      /// A typedef's anonymous tag declaration.  Identifier is empty.
      TypedefAnonDecl,

      /// An Objective-C interface.
      ObjCInterface,

      /// An Objective-C protocol.
      ObjCProtocol,
    };

    static bool requiresIdentifier(ComponentKind kind) {
      return kind != TypedefAnonDecl;
    }

    SmallVector<std::pair<ComponentKind, Identifier>, 2> Path;

    void add(ComponentKind kind, Identifier name) {
      Path.push_back({kind, name});
    }
  };
private:
  TaggedUnion<void, const Decl *, ExternalPath> Union;

public:
  StableSerializationPath() {}
  StableSerializationPath(const Decl *d) : Union(d) {}
  StableSerializationPath(ExternalPath ext) : Union(ext) {}

  explicit operator bool() const { return !Union.empty(); }

  bool isCodiraDecl() const { return Union.isa<const Decl*>(); }
  const Decl *getCodiraDecl() const {
    assert(isCodiraDecl());
    return Union.get<const Decl*>();
  }

  bool isExternalPath() const { return Union.isa<ExternalPath>(); }
  const ExternalPath &getExternalPath() const {
    assert(isExternalPath());
    return Union.get<ExternalPath>();
  }

  LANGUAGE_DEBUG_DUMP;
  void dump(raw_ostream &os) const;
};

class ClangModuleLoader : public ModuleLoader {
private:
  virtual void anchor() override;

protected:
  using ModuleLoader::ModuleLoader;

public:
  /// This module loader's Clang instance may be configured with a different
  /// (higher) OS version than the compilation target itself in order to be able
  /// to load pre-compiled Clang modules that are aligned with the broader SDK,
  /// and match the SDK deployment target against which Codira modules are also
  /// built.
  ///
  /// In this case, we must use the Codira compiler's OS version triple when
  /// performing codegen, and the importer's Clang instance OS version triple
  /// during module loading. `getModuleAvailabilityTarget` is for module-loading
  /// clients only, and uses the latter.
  ///
  /// (The implementing `ClangImporter` class maintains separate Target info
  /// for use by IRGen/CodeGen clients)
  virtual language::Core::TargetInfo &getModuleAvailabilityTarget() const = 0;

  virtual language::Core::ASTContext &getClangASTContext() const = 0;
  virtual language::Core::Preprocessor &getClangPreprocessor() const = 0;
  virtual language::Core::Sema &getClangSema() const = 0;
  virtual const language::Core::CompilerInstance &getClangInstance() const = 0;
  virtual void printStatistics() const = 0;
  virtual void dumpCodiraLookupTables() const = 0;

  /// Returns the module that contains imports and declarations from all loaded
  /// header files.
  virtual ModuleDecl *getImportedHeaderModule() const = 0;

  /// Retrieves the Codira wrapper for the given Clang module, creating
  /// it if necessary.
  virtual ModuleDecl *
  getWrapperForModule(const language::Core::Module *mod,
                      bool returnOverlayIfPossible = false) const = 0;

  /// Adds a new search path to the Clang CompilerInstance, as if specified with
  /// -I or -F.
  ///
  /// \returns true if there was an error adding the search path.
  virtual bool addSearchPath(StringRef newSearchPath, bool isFramework,
                             bool isSystem) = 0;

  /// Determine whether \c overlayDC is within an overlay module for the
  /// imported context enclosing \c importedDC.
  ///
  /// This routine is used for various hacks that are only permitted within
  /// overlays of imported modules, e.g., Objective-C bridging conformances.
  virtual bool
  isInOverlayModuleForImportedModule(const DeclContext *overlayDC,
                                     const DeclContext *importedDC) = 0;

  /// Look for declarations associated with the given name.
  ///
  /// \param name The name we're searching for.
  virtual void lookupValue(DeclName name, VisibleDeclConsumer &consumer) = 0;

  /// Look up a type declaration by its Clang name.
  ///
  /// Note that this method does no filtering. If it finds the type in a loaded
  /// module, it returns it. This is intended for use in reflection / debugging
  /// contexts where access is not a problem.
  virtual void
  lookupTypeDecl(StringRef clangName, ClangTypeKind kind,
                 toolchain::function_ref<void(TypeDecl *)> receiver) = 0;

  /// Look up type a declaration synthesized by the Clang importer itself, using
  /// a "related entity kind" to determine which type it should be. For example,
  /// this can be used to find the synthesized error struct for an
  /// NS_ERROR_ENUM.
  ///
  /// Note that this method does no filtering. If it finds the type in a loaded
  /// module, it returns it. This is intended for use in reflection / debugging
  /// contexts where access is not a problem.
  virtual void
  lookupRelatedEntity(StringRef clangName, ClangTypeKind kind,
                      StringRef relatedEntityKind,
                      toolchain::function_ref<void(TypeDecl *)> receiver) = 0;

  /// Imports a clang decl directly, rather than looking up its name.
  virtual Decl *importDeclDirectly(const language::Core::NamedDecl *decl) = 0;

  /// Clones an imported \param decl from its base class to its derived class
  /// \param newContext where it is inherited. Its access level is determined
  /// with respect to \param inheritance, which signifies whether \param decl
  /// was inherited via C++ public/protected/private inheritance.
  ///
  /// This function uses a cache so that it is idempotent; successive
  /// invocations will only generate one cloned ValueDecl (and all return
  /// a pointer to it). Returns a NULL pointer upon failure.
  virtual ValueDecl *importBaseMemberDecl(ValueDecl *decl,
                                          DeclContext *newContext,
                                          ClangInheritanceInfo inheritance) = 0;

  /// Returnes the original method if \param decl is a clone from a base class
  virtual ValueDecl *getOriginalForClonedMember(const ValueDecl *decl) = 0;

  /// Emits diagnostics for any declarations named name
  /// whose direct declaration context is a TU.
  virtual void diagnoseTopLevelValue(const DeclName &name) = 0;

  /// Emit diagnostics for declarations named name that are members
  /// of the provided baseType.
  virtual void diagnoseMemberValue(const DeclName &name,
                                   const Type &baseType) = 0;

  /// Instantiate and import class template using given arguments.
  ///
  /// This method will find the language::Core::ClassTemplateSpecialization decl if
  /// it already exists, or it will create one. Then it will import this
  /// decl the same way as we import typedeffed class templates - using
  /// the hidden struct prefixed with `__CxxTemplateInst`.
  virtual StructDecl *
  instantiateCXXClassTemplate(language::Core::ClassTemplateDecl *decl,
                      ArrayRef<language::Core::TemplateArgument> arguments) = 0;

  virtual ConcreteDeclRef
  getCXXFunctionTemplateSpecialization(SubstitutionMap subst,
                                       ValueDecl *decl) = 0;

  /// Try to parse the string as a Clang function type.
  ///
  /// Returns null if there was a parsing failure.
  virtual const language::Core::Type *parseClangFunctionType(StringRef type,
                                                    SourceLoc loc) const = 0;

  /// Print the Clang type.
  virtual void printClangType(const language::Core::Type *type,
                              toolchain::raw_ostream &os) const = 0;

  /// Try to find a stable serialization path for the given declaration,
  /// if there is one.
  virtual StableSerializationPath
  findStableSerializationPath(const language::Core::Decl *decl) const = 0;

  /// Try to resolve a stable serialization path down to the original
  /// declaration.
  virtual const language::Core::Decl *
  resolveStableSerializationPath(const StableSerializationPath &path) const = 0;

  /// Determine whether the given type is serializable.
  ///
  /// If \c checkCanonical is true, checks the canonical type,
  /// not the given possibly-sugared type.  In general:
  ///  - non-canonical representations should be preserving the
  ///    sugared type even if it isn't serializable, since that
  ///    maintains greater source fidelity;
  ///  - semantic checks need to be checking the serializability
  ///    of the canonical type, since it's always potentially
  ///    necessary to serialize that (e.g. in SIL); and
  ///  - serializers can try to serialize the sugared type to
  ///    maintain source fidelity and just fall back on the canonical
  ///    type if that's not possible.
  ///
  /// The expectation here is that this predicate is meaningful
  /// independent of the actual form of serialization: the types
  /// that we can't reliably binary-serialize without an absolute
  /// Clang AST cross-reference are the same types that won't
  /// reliably round-trip through a textual format.  At the very
  /// least, it's probably best to use conservative predicates
  /// that work both ways so that language behavior doesn't differ
  /// based on subtleties like the target module interface format.
  virtual bool isSerializable(const language::Core::Type *type,
                              bool checkCanonical) const = 0;

  virtual language::Core::FunctionDecl *
  instantiateCXXFunctionTemplate(ASTContext &ctx,
                                 language::Core::FunctionTemplateDecl *fn,
                                 SubstitutionMap subst) = 0;

  virtual bool isCXXMethodMutating(const language::Core::CXXMethodDecl *method) = 0;

  virtual bool isUnsafeCXXMethod(const FuncDecl *fn) = 0;

  virtual FuncDecl *getDefaultArgGenerator(const language::Core::ParmVarDecl *param) = 0;

  virtual FuncDecl *
  getAvailabilityDomainPredicate(const language::Core::VarDecl *var) = 0;

  virtual std::optional<Type>
  importFunctionReturnType(const language::Core::FunctionDecl *clangDecl,
                           DeclContext *dc) = 0;

  virtual Type importVarDeclType(const language::Core::VarDecl *clangDecl,
                                 VarDecl *languageDecl,
                                 DeclContext *dc) = 0;

  /// Find the lookup table that corresponds to the given Clang module.
  ///
  /// \param clangModule The module, or null to indicate that we're talking
  /// about the directly-parsed headers.
  virtual CodiraLookupTable *
  findLookupTable(const language::Core::Module *clangModule) = 0;

  virtual DeclName
  importName(const language::Core::NamedDecl *D,
             language::Core::DeclarationName givenName = language::Core::DeclarationName()) = 0;

  /// Determine the effective Clang context for the given Codira nominal type.
  virtual EffectiveClangContext getEffectiveClangContext(
      const NominalTypeDecl *nominal) = 0;

  virtual const language::Core::TypedefType *
  getTypeDefForCXXCFOptionsDefinition(const language::Core::Decl *candidateDecl) = 0;

  virtual SourceLoc importSourceLocation(language::Core::SourceLocation loc) = 0;

  /// Just like Decl::getClangNode() except we look through to the 'Code'
  /// enum of an error wrapper struct.
  virtual ClangNode getEffectiveClangNode(const Decl *decl) const = 0;
};

/// Describes a C++ template instantiation error.
struct TemplateInstantiationError {
  /// Generic types that could not be converted to QualTypes using the
  /// ClangTypeConverter.
  SmallVector<Type, 4> failedTypes;
};

} // namespace language

#endif // TOOLCHAIN_LANGUAGE_AST_CLANG_MODULE_LOADER_H
