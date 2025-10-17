/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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

//===- ParsedAttrInfo.h - Info needed to parse an attribute -----*- C++ -*-===//
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
// This file defines the ParsedAttrInfo class, which dictates how to
// parse an attribute. This class is the one that plugins derive to
// define a new attribute.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_CORE_BASIC_PARSEDATTRINFO_H
#define LANGUAGE_CORE_BASIC_PARSEDATTRINFO_H

#include "language/Core/Basic/AttrSubjectMatchRules.h"
#include "language/Core/Basic/AttributeCommonInfo.h"
#include "language/Core/Support/Compiler.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/Support/Registry.h"
#include <climits>
#include <list>

namespace language::Core {

class Attr;
class Decl;
class LangOptions;
class ParsedAttr;
class Sema;
class Stmt;
class TargetInfo;

struct ParsedAttrInfo {
  /// Corresponds to the Kind enum.
  LLVM_PREFERRED_TYPE(AttributeCommonInfo::Kind)
  unsigned AttrKind : 16;
  /// The number of required arguments of this attribute.
  unsigned NumArgs : 4;
  /// The number of optional arguments of this attributes.
  unsigned OptArgs : 4;
  /// The number of non-fake arguments specified in the attribute definition.
  unsigned NumArgMembers : 4;
  /// True if the parsing does not match the semantic content.
  LLVM_PREFERRED_TYPE(bool)
  unsigned HasCustomParsing : 1;
  // True if this attribute accepts expression parameter pack expansions.
  LLVM_PREFERRED_TYPE(bool)
  unsigned AcceptsExprPack : 1;
  /// True if this attribute is only available for certain targets.
  LLVM_PREFERRED_TYPE(bool)
  unsigned IsTargetSpecific : 1;
  /// True if this attribute applies to types.
  LLVM_PREFERRED_TYPE(bool)
  unsigned IsType : 1;
  /// True if this attribute applies to statements.
  LLVM_PREFERRED_TYPE(bool)
  unsigned IsStmt : 1;
  /// True if this attribute has any spellings that are known to gcc.
  LLVM_PREFERRED_TYPE(bool)
  unsigned IsKnownToGCC : 1;
  /// True if this attribute is supported by #pragma clang attribute.
  LLVM_PREFERRED_TYPE(bool)
  unsigned IsSupportedByPragmaAttribute : 1;
  /// The syntaxes supported by this attribute and how they're spelled.
  struct Spelling {
    AttributeCommonInfo::Syntax Syntax;
    const char *NormalizedFullName;
  };
  ArrayRef<Spelling> Spellings;
  // The names of the known arguments of this attribute.
  ArrayRef<const char *> ArgNames;

protected:
  constexpr ParsedAttrInfo(AttributeCommonInfo::Kind AttrKind =
                               AttributeCommonInfo::NoSemaHandlerAttribute)
      : AttrKind(AttrKind), NumArgs(0), OptArgs(0), NumArgMembers(0),
        HasCustomParsing(0), AcceptsExprPack(0), IsTargetSpecific(0), IsType(0),
        IsStmt(0), IsKnownToGCC(0), IsSupportedByPragmaAttribute(0) {}

  constexpr ParsedAttrInfo(AttributeCommonInfo::Kind AttrKind, unsigned NumArgs,
                           unsigned OptArgs, unsigned NumArgMembers,
                           unsigned HasCustomParsing, unsigned AcceptsExprPack,
                           unsigned IsTargetSpecific, unsigned IsType,
                           unsigned IsStmt, unsigned IsKnownToGCC,
                           unsigned IsSupportedByPragmaAttribute,
                           ArrayRef<Spelling> Spellings,
                           ArrayRef<const char *> ArgNames)
      : AttrKind(AttrKind), NumArgs(NumArgs), OptArgs(OptArgs),
        NumArgMembers(NumArgMembers), HasCustomParsing(HasCustomParsing),
        AcceptsExprPack(AcceptsExprPack), IsTargetSpecific(IsTargetSpecific),
        IsType(IsType), IsStmt(IsStmt), IsKnownToGCC(IsKnownToGCC),
        IsSupportedByPragmaAttribute(IsSupportedByPragmaAttribute),
        Spellings(Spellings), ArgNames(ArgNames) {}

public:
  virtual ~ParsedAttrInfo() = default;

  /// Check if this attribute has specified spelling.
  bool hasSpelling(AttributeCommonInfo::Syntax Syntax, StringRef Name) const {
    return toolchain::any_of(Spellings, [&](const Spelling &S) {
      return (S.Syntax == Syntax && S.NormalizedFullName == Name);
    });
  }

  /// Check if this attribute appertains to D, and issue a diagnostic if not.
  virtual bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                                    const Decl *D) const {
    return true;
  }
  /// Check if this attribute appertains to St, and issue a diagnostic if not.
  virtual bool diagAppertainsToStmt(Sema &S, const ParsedAttr &Attr,
                                    const Stmt *St) const {
    return true;
  }
  /// Check if the given attribute is mutually exclusive with other attributes
  /// already applied to the given declaration.
  virtual bool diagMutualExclusion(Sema &S, const ParsedAttr &A,
                                   const Decl *D) const {
    return true;
  }
  /// Check if this attribute is allowed by the language we are compiling.
  virtual bool acceptsLangOpts(const LangOptions &LO) const { return true; }

  /// Check if this attribute is allowed when compiling for the given target.
  virtual bool existsInTarget(const TargetInfo &Target) const { return true; }

  /// Check if this attribute's spelling is allowed when compiling for the given
  /// target.
  virtual bool spellingExistsInTarget(const TargetInfo &Target,
                                      const unsigned SpellingListIndex) const {
    return true;
  }

  /// Convert the spelling index of Attr to a semantic spelling enum value.
  virtual unsigned
  spellingIndexToSemanticSpelling(const ParsedAttr &Attr) const {
    return UINT_MAX;
  }
  /// Returns true if the specified parameter index for this attribute in
  /// Attr.td is an ExprArgument or VariadicExprArgument, or a subclass thereof;
  /// returns false otherwise.
  virtual bool isParamExpr(size_t N) const { return false; }
  /// Populate Rules with the match rules of this attribute.
  virtual void getPragmaAttributeMatchRules(
      toolchain::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &Rules,
      const LangOptions &LangOpts) const {}

  enum AttrHandling { NotHandled, AttributeApplied, AttributeNotApplied };
  /// If this ParsedAttrInfo knows how to handle this ParsedAttr applied to this
  /// Decl then do so and return either AttributeApplied if it was applied or
  /// AttributeNotApplied if it wasn't. Otherwise return NotHandled.
  virtual AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                           const ParsedAttr &Attr) const {
    return NotHandled;
  }
  /// If this ParsedAttrInfo knows how to handle this ParsedAttr applied to this
  /// Stmt then do so (referencing the resulting Attr in Result) and return
  /// either AttributeApplied if it was applied or AttributeNotApplied if it
  /// wasn't. Otherwise return NotHandled.
  virtual AttrHandling handleStmtAttribute(Sema &S, Stmt *St,
                                           const ParsedAttr &Attr,
                                           class Attr *&Result) const {
    return NotHandled;
  }

  static const ParsedAttrInfo &get(const AttributeCommonInfo &A);
  static ArrayRef<const ParsedAttrInfo *> getAllBuiltin();
};

typedef toolchain::Registry<ParsedAttrInfo> ParsedAttrInfoRegistry;

const std::list<std::unique_ptr<ParsedAttrInfo>> &getAttributePluginInstances();

} // namespace language::Core

namespace toolchain {
extern template class CLANG_TEMPLATE_ABI Registry<language::Core::ParsedAttrInfo>;
} // namespace toolchain

#endif // LANGUAGE_CORE_BASIC_PARSEDATTRINFO_H
