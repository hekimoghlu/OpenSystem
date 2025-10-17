/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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

//===--- StandardLibrary.cpp ------------------------------------*- C++ -*-===//
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

#include "language/Core/Tooling/Inclusions/StandardLibrary.h"
#include "language/Core/AST/Decl.h"
#include "language/Core/Basic/LangOptions.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/DenseSet.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include <optional>

namespace language::Core {
namespace tooling {
namespace stdlib {

namespace {
// Symbol name -> Symbol::ID, within a namespace.
using NSSymbolMap = toolchain::DenseMap<toolchain::StringRef, unsigned>;

// A Mapping per language.
struct SymbolHeaderMapping {
  toolchain::StringRef *HeaderNames = nullptr;
  // Header name => Header::ID
  toolchain::DenseMap<toolchain::StringRef, unsigned> *HeaderIDs;

  unsigned SymbolCount = 0;
  // Symbol::ID => symbol qualified_name/name/scope
  struct SymbolName {
    const char *Data;  // std::vector
    unsigned ScopeLen; // ~~~~~
    unsigned NameLen;  //      ~~~~~~
    StringRef scope() const { return StringRef(Data, ScopeLen); }
    StringRef name() const { return StringRef(Data + ScopeLen, NameLen); }
    StringRef qualifiedName() const {
      return StringRef(Data, ScopeLen + NameLen);
    }
  } *SymbolNames = nullptr;
  // Symbol name -> Symbol::ID, within a namespace.
  toolchain::DenseMap<toolchain::StringRef, NSSymbolMap *> *NamespaceSymbols = nullptr;
  // Symbol::ID => Header::ID
  toolchain::SmallVector<unsigned> *SymbolHeaderIDs = nullptr;
};
} // namespace
static SymbolHeaderMapping
    *LanguageMappings[static_cast<unsigned>(Lang::LastValue) + 1];
static const SymbolHeaderMapping *getMappingPerLang(Lang L) {
  return LanguageMappings[static_cast<unsigned>(L)];
}

static int countSymbols(Lang Language) {
  ArrayRef<const char *> Symbols;
#define SYMBOL(Name, NS, Header) #NS #Name,
  switch (Language) {
  case Lang::C: {
    static constexpr const char *CSymbols[] = {
#include "CSpecialSymbolMap.inc"
#include "CSymbolMap.inc"
    };
    Symbols = CSymbols;
    break;
  }
  case Lang::CXX: {
    static constexpr const char *CXXSymbols[] = {
#include "StdSpecialSymbolMap.inc"
#include "StdSymbolMap.inc"
#include "StdTsSymbolMap.inc"
    };
    Symbols = CXXSymbols;
    break;
  }
  }
#undef SYMBOL
  return toolchain::DenseSet<StringRef>(toolchain::from_range, Symbols).size();
}

static int initialize(Lang Language) {
  SymbolHeaderMapping *Mapping = new SymbolHeaderMapping();
  LanguageMappings[static_cast<unsigned>(Language)] = Mapping;

  unsigned SymCount = countSymbols(Language);
  Mapping->SymbolCount = SymCount;
  Mapping->SymbolNames =
      new std::remove_reference_t<decltype(*Mapping->SymbolNames)>[SymCount];
  Mapping->SymbolHeaderIDs = new std::remove_reference_t<
      decltype(*Mapping->SymbolHeaderIDs)>[SymCount];
  Mapping->NamespaceSymbols =
      new std::remove_reference_t<decltype(*Mapping->NamespaceSymbols)>;
  Mapping->HeaderIDs =
      new std::remove_reference_t<decltype(*Mapping->HeaderIDs)>;
  auto AddNS = [&](toolchain::StringRef NS) -> NSSymbolMap & {
    auto R = Mapping->NamespaceSymbols->try_emplace(NS, nullptr);
    if (R.second)
      R.first->second = new NSSymbolMap();
    return *R.first->second;
  };

  auto AddHeader = [&](toolchain::StringRef Header) -> unsigned {
    return Mapping->HeaderIDs->try_emplace(Header, Mapping->HeaderIDs->size())
        .first->second;
  };

  auto Add = [&, SymIndex(-1)](toolchain::StringRef QName, unsigned NSLen,
                               toolchain::StringRef HeaderName) mutable {
    // Correct "Nonefoo" => foo.
    // FIXME: get rid of "None" from the generated mapping files.
    if (QName.take_front(NSLen) == "None") {
      QName = QName.drop_front(NSLen);
      NSLen = 0;
    }

    if (SymIndex > 0) {
      assert(toolchain::none_of(toolchain::ArrayRef(Mapping->SymbolNames, SymIndex),
                           [&QName](const SymbolHeaderMapping::SymbolName &S) {
                             return S.qualifiedName() == QName;
                           }) &&
             "The symbol has been added before, make sure entries in the .inc "
             "file are grouped by symbol name!");
    }
    if (SymIndex < 0 ||
        Mapping->SymbolNames[SymIndex].qualifiedName() != QName) {
      // First symbol or new symbol, increment next available index.
      ++SymIndex;
    } // Else use the same index.
    Mapping->SymbolNames[SymIndex] = {
        QName.data(), NSLen, static_cast<unsigned int>(QName.size() - NSLen)};
    if (!HeaderName.empty())
      Mapping->SymbolHeaderIDs[SymIndex].push_back(AddHeader(HeaderName));

    NSSymbolMap &NSSymbols = AddNS(QName.take_front(NSLen));
    NSSymbols.try_emplace(QName.drop_front(NSLen), SymIndex);
  };

  struct Symbol {
    const char *QName;
    unsigned NSLen;
    const char *HeaderName;
  };
#define SYMBOL(Name, NS, Header)                                               \
  {#NS #Name, static_cast<decltype(Symbol::NSLen)>(StringRef(#NS).size()),     \
   #Header},
  switch (Language) {
  case Lang::C: {
    static constexpr Symbol CSymbols[] = {
#include "CSpecialSymbolMap.inc"
#include "CSymbolMap.inc"
    };
    for (const Symbol &S : CSymbols)
      Add(S.QName, S.NSLen, S.HeaderName);
    break;
  }
  case Lang::CXX: {
    static constexpr Symbol CXXSymbols[] = {
#include "StdSpecialSymbolMap.inc"
#include "StdSymbolMap.inc"
#include "StdTsSymbolMap.inc"
    };
    for (const Symbol &S : CXXSymbols)
      Add(S.QName, S.NSLen, S.HeaderName);
    break;
  }
  }
#undef SYMBOL

  Mapping->HeaderNames = new toolchain::StringRef[Mapping->HeaderIDs->size()];
  for (const auto &E : *Mapping->HeaderIDs)
    Mapping->HeaderNames[E.second] = E.first;

  return 0;
}

static void ensureInitialized() {
  static int Dummy = []() {
    for (unsigned L = 0; L <= static_cast<unsigned>(Lang::LastValue); ++L)
      initialize(static_cast<Lang>(L));
    return 0;
  }();
  (void)Dummy;
}

std::vector<Header> Header::all(Lang L) {
  ensureInitialized();
  std::vector<Header> Result;
  const auto *Mapping = getMappingPerLang(L);
  Result.reserve(Mapping->HeaderIDs->size());
  for (unsigned I = 0, E = Mapping->HeaderIDs->size(); I < E; ++I)
    Result.push_back(Header(I, L));
  return Result;
}
std::optional<Header> Header::named(toolchain::StringRef Name, Lang L) {
  ensureInitialized();
  const auto *Mapping = getMappingPerLang(L);
  auto It = Mapping->HeaderIDs->find(Name);
  if (It == Mapping->HeaderIDs->end())
    return std::nullopt;
  return Header(It->second, L);
}
toolchain::StringRef Header::name() const {
  return getMappingPerLang(Language)->HeaderNames[ID];
}

std::vector<Symbol> Symbol::all(Lang L) {
  ensureInitialized();
  std::vector<Symbol> Result;
  const auto *Mapping = getMappingPerLang(L);
  Result.reserve(Mapping->SymbolCount);
  for (unsigned I = 0, E = Mapping->SymbolCount; I < E; ++I)
    Result.push_back(Symbol(I, L));
  return Result;
}
toolchain::StringRef Symbol::scope() const {
  return getMappingPerLang(Language)->SymbolNames[ID].scope();
}
toolchain::StringRef Symbol::name() const {
  return getMappingPerLang(Language)->SymbolNames[ID].name();
}
toolchain::StringRef Symbol::qualifiedName() const {
  return getMappingPerLang(Language)->SymbolNames[ID].qualifiedName();
}
std::optional<Symbol> Symbol::named(toolchain::StringRef Scope, toolchain::StringRef Name,
                                    Lang L) {
  ensureInitialized();

  if (NSSymbolMap *NSSymbols =
          getMappingPerLang(L)->NamespaceSymbols->lookup(Scope)) {
    auto It = NSSymbols->find(Name);
    if (It != NSSymbols->end())
      return Symbol(It->second, L);
  }
  return std::nullopt;
}
std::optional<Header> Symbol::header() const {
  const auto &Headers = getMappingPerLang(Language)->SymbolHeaderIDs[ID];
  if (Headers.empty())
    return std::nullopt;
  return Header(Headers.front(), Language);
}
toolchain::SmallVector<Header> Symbol::headers() const {
  toolchain::SmallVector<Header> Results;
  for (auto HeaderID : getMappingPerLang(Language)->SymbolHeaderIDs[ID])
    Results.emplace_back(Header(HeaderID, Language));
  return Results;
}

Recognizer::Recognizer() { ensureInitialized(); }

NSSymbolMap *Recognizer::namespaceSymbols(const DeclContext *DC, Lang L) {
  if (DC->isTranslationUnit()) // global scope.
    return getMappingPerLang(L)->NamespaceSymbols->lookup("");

  auto It = NamespaceCache.find(DC);
  if (It != NamespaceCache.end())
    return It->second;
  const NamespaceDecl *D = toolchain::cast<NamespaceDecl>(DC);
  NSSymbolMap *Result = [&]() -> NSSymbolMap * {
    if (D->isAnonymousNamespace())
      return nullptr;
    // Print the namespace and its parents ommitting inline scopes.
    std::string Scope;
    for (const auto *ND = D; ND;
         ND = toolchain::dyn_cast_or_null<NamespaceDecl>(ND->getParent()))
      if (!ND->isInlineNamespace() && !ND->isAnonymousNamespace())
        Scope = ND->getName().str() + "::" + Scope;
    return getMappingPerLang(L)->NamespaceSymbols->lookup(Scope);
  }();
  NamespaceCache.try_emplace(D, Result);
  return Result;
}

std::optional<Symbol> Recognizer::operator()(const Decl *D) {
  Lang L;
  if (D->getLangOpts().CPlusPlus)
    L = Lang::CXX;
  else if (D->getLangOpts().C99)
    L = Lang::C;
  else
    return std::nullopt; // not a supported language.

  // If D is std::vector::iterator, `vector` is the outer symbol to look up.
  // We keep all the candidate DCs as some may turn out to be anon enums.
  // Do this resolution lazily as we may turn out not to have a std namespace.
  toolchain::SmallVector<const DeclContext *> IntermediateDecl;
  const DeclContext *DC = D->getDeclContext();
  if (!DC) // The passed D is a TranslationUnitDecl!
    return std::nullopt;
  while (!DC->isNamespace() && !DC->isTranslationUnit()) {
    if (NamedDecl::classofKind(DC->getDeclKind()))
      IntermediateDecl.push_back(DC);
    DC = DC->getParent();
  }
  NSSymbolMap *Symbols = namespaceSymbols(DC, L);
  if (!Symbols)
    return std::nullopt;

  toolchain::StringRef Name = [&]() -> toolchain::StringRef {
    for (const auto *SymDC : toolchain::reverse(IntermediateDecl)) {
      DeclarationName N = cast<NamedDecl>(SymDC)->getDeclName();
      if (const auto *II = N.getAsIdentifierInfo())
        return II->getName();
      if (!N.isEmpty())
        return ""; // e.g. operator<: give up
    }
    if (const auto *ND = toolchain::dyn_cast<NamedDecl>(D))
      if (const auto *II = ND->getIdentifier())
        return II->getName();
    return "";
  }();
  if (Name.empty())
    return std::nullopt;

  auto It = Symbols->find(Name);
  if (It == Symbols->end())
    return std::nullopt;
  return Symbol(It->second, L);
}

} // namespace stdlib
} // namespace tooling
} // namespace language::Core
