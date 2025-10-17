/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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

//===- IndexingAction.cpp - Frontend index action -------------------------===//
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

#include "clang/Index/IndexingAction.h"
#include "IndexingContext.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Serialization/ASTReader.h"
#include <memory>

using namespace language::Core;
using namespace language::Core::index;

namespace {

class IndexPPCallbacks final : public PPCallbacks {
  std::shared_ptr<IndexingContext> IndexCtx;

public:
  IndexPPCallbacks(std::shared_ptr<IndexingContext> IndexCtx)
      : IndexCtx(std::move(IndexCtx)) {}

  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override {
    IndexCtx->handleMacroReference(*MacroNameTok.getIdentifierInfo(),
                                   Range.getBegin(), *MD.getMacroInfo());
  }

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    IndexCtx->handleMacroDefined(*MacroNameTok.getIdentifierInfo(),
                                 MacroNameTok.getLocation(),
                                 *MD->getMacroInfo());
  }

  void MacroUndefined(const Token &MacroNameTok, const MacroDefinition &MD,
                      const MacroDirective *Undef) override {
    if (!MD.getMacroInfo())  // Ignore noop #undef.
      return;
    IndexCtx->handleMacroUndefined(*MacroNameTok.getIdentifierInfo(),
                                   MacroNameTok.getLocation(),
                                   *MD.getMacroInfo());
  }

  void Defined(const Token &MacroNameTok, const MacroDefinition &MD,
               SourceRange Range) override {
    if (!MD.getMacroInfo()) // Ignore nonexistent macro.
      return;
    // Note: this is defined(M), not #define M
    IndexCtx->handleMacroReference(*MacroNameTok.getIdentifierInfo(),
                                   MacroNameTok.getLocation(),
                                   *MD.getMacroInfo());
  }
  void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
             const MacroDefinition &MD) override {
    if (!MD.getMacroInfo()) // Ignore non-existent macro.
      return;
    IndexCtx->handleMacroReference(*MacroNameTok.getIdentifierInfo(),
                                   MacroNameTok.getLocation(),
                                   *MD.getMacroInfo());
  }
  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MD) override {
    if (!MD.getMacroInfo()) // Ignore nonexistent macro.
      return;
    IndexCtx->handleMacroReference(*MacroNameTok.getIdentifierInfo(),
                                   MacroNameTok.getLocation(),
                                   *MD.getMacroInfo());
  }

  using PPCallbacks::Elifdef;
  using PPCallbacks::Elifndef;
  void Elifdef(SourceLocation Loc, const Token &MacroNameTok,
               const MacroDefinition &MD) override {
    if (!MD.getMacroInfo()) // Ignore non-existent macro.
      return;
    IndexCtx->handleMacroReference(*MacroNameTok.getIdentifierInfo(),
                                   MacroNameTok.getLocation(),
                                   *MD.getMacroInfo());
  }
  void Elifndef(SourceLocation Loc, const Token &MacroNameTok,
                const MacroDefinition &MD) override {
    if (!MD.getMacroInfo()) // Ignore non-existent macro.
      return;
    IndexCtx->handleMacroReference(*MacroNameTok.getIdentifierInfo(),
                                   MacroNameTok.getLocation(),
                                   *MD.getMacroInfo());
  }
};

class IndexASTConsumer final : public ASTConsumer {
  std::shared_ptr<IndexDataConsumer> DataConsumer;
  std::shared_ptr<IndexingContext> IndexCtx;
  std::shared_ptr<Preprocessor> PP;
  std::function<bool(const Decl *)> ShouldSkipFunctionBody;

public:
  IndexASTConsumer(std::shared_ptr<IndexDataConsumer> DataConsumer,
                   const IndexingOptions &Opts,
                   std::shared_ptr<Preprocessor> PP,
                   std::function<bool(const Decl *)> ShouldSkipFunctionBody)
      : DataConsumer(std::move(DataConsumer)),
        IndexCtx(new IndexingContext(Opts, *this->DataConsumer)),
        PP(std::move(PP)),
        ShouldSkipFunctionBody(std::move(ShouldSkipFunctionBody)) {
    assert(this->DataConsumer != nullptr);
    assert(this->PP != nullptr);
  }

protected:
  void Initialize(ASTContext &Context) override {
    IndexCtx->setASTContext(Context);
    IndexCtx->getDataConsumer().initialize(Context);
    IndexCtx->getDataConsumer().setPreprocessor(PP);
    PP->addPPCallbacks(std::make_unique<IndexPPCallbacks>(IndexCtx));
  }

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    return IndexCtx->indexDeclGroupRef(DG);
  }

  void HandleInterestingDecl(DeclGroupRef DG) override {
    // Ignore deserialized decls.
  }

  void HandleTopLevelDeclInObjCContainer(DeclGroupRef DG) override {
    IndexCtx->indexDeclGroupRef(DG);
  }

  void HandleTranslationUnit(ASTContext &Ctx) override {
    DataConsumer->finish();
  }

  bool shouldSkipFunctionBody(Decl *D) override {
    return ShouldSkipFunctionBody(D);
  }
};

class IndexAction final : public ASTFrontendAction {
  std::shared_ptr<IndexDataConsumer> DataConsumer;
  IndexingOptions Opts;

public:
  IndexAction(std::shared_ptr<IndexDataConsumer> DataConsumer,
              const IndexingOptions &Opts)
      : DataConsumer(std::move(DataConsumer)), Opts(Opts) {
    assert(this->DataConsumer != nullptr);
  }

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<IndexASTConsumer>(
        DataConsumer, Opts, CI.getPreprocessorPtr(),
        /*ShouldSkipFunctionBody=*/[](const Decl *) { return false; });
  }
};

} // anonymous namespace

std::unique_ptr<ASTConsumer> index::createIndexingASTConsumer(
    std::shared_ptr<IndexDataConsumer> DataConsumer,
    const IndexingOptions &Opts, std::shared_ptr<Preprocessor> PP,
    std::function<bool(const Decl *)> ShouldSkipFunctionBody) {
  return std::make_unique<IndexASTConsumer>(DataConsumer, Opts, PP,
                                            ShouldSkipFunctionBody);
}

std::unique_ptr<ASTConsumer> clang::index::createIndexingASTConsumer(
    std::shared_ptr<IndexDataConsumer> DataConsumer,
    const IndexingOptions &Opts, std::shared_ptr<Preprocessor> PP) {
  std::function<bool(const Decl *)> ShouldSkipFunctionBody = [](const Decl *) {
    return false;
  };
  if (Opts.ShouldTraverseDecl)
    ShouldSkipFunctionBody =
        [ShouldTraverseDecl(Opts.ShouldTraverseDecl)](const Decl *D) {
          return !ShouldTraverseDecl(D);
        };
  return createIndexingASTConsumer(std::move(DataConsumer), Opts, std::move(PP),
                                   std::move(ShouldSkipFunctionBody));
}

std::unique_ptr<FrontendAction>
index::createIndexingAction(std::shared_ptr<IndexDataConsumer> DataConsumer,
                            const IndexingOptions &Opts) {
  assert(DataConsumer != nullptr);
  return std::make_unique<IndexAction>(std::move(DataConsumer), Opts);
}

static bool topLevelDeclVisitor(void *context, const Decl *D) {
  IndexingContext &IndexCtx = *static_cast<IndexingContext *>(context);
  return IndexCtx.indexTopLevelDecl(D);
}

static void indexTranslationUnit(ASTUnit &Unit, IndexingContext &IndexCtx) {
  Unit.visitLocalTopLevelDecls(&IndexCtx, topLevelDeclVisitor);
}

static void indexPreprocessorMacro(const IdentifierInfo *II,
                                   const MacroInfo *MI,
                                   MacroDirective::Kind DirectiveKind,
                                   SourceLocation Loc,
                                   IndexDataConsumer &DataConsumer) {
  // When using modules, it may happen that we find #undef of a macro that
  // was defined in another module. In such case, MI may be nullptr, since
  // we only look for macro definitions in the current TU. In that case,
  // there is nothing to index.
  if (!MI)
    return;

  // Skip implicit visibility change.
  if (DirectiveKind == MacroDirective::MD_Visibility)
    return;

  auto Role = DirectiveKind == MacroDirective::MD_Define
                  ? SymbolRole::Definition
                  : SymbolRole::Undefinition;
  DataConsumer.handleMacroOccurrence(II, MI, static_cast<unsigned>(Role), Loc);
}

static void indexPreprocessorMacros(Preprocessor &PP,
                                    IndexDataConsumer &DataConsumer) {
  for (const auto &M : PP.macros()) {
    for (auto *MD = M.second.getLatest(); MD; MD = MD->getPrevious()) {
      indexPreprocessorMacro(M.first, MD->getMacroInfo(), MD->getKind(),
                             MD->getLocation(), DataConsumer);
    }
  }
}

static void indexPreprocessorModuleMacros(Preprocessor &PP,
                                          serialization::ModuleFile &Mod,
                                          IndexDataConsumer &DataConsumer) {
  for (const auto &M : PP.macros()) {
    if (M.second.getLatest() == nullptr) {
      for (auto *MM : PP.getLeafModuleMacros(M.first)) {
        auto *OwningMod = MM->getOwningModule();
        if (OwningMod && OwningMod->getASTFile() == Mod.File) {
          if (auto *MI = MM->getMacroInfo()) {
            indexPreprocessorMacro(M.first, MI, MacroDirective::MD_Define,
                                   MI->getDefinitionLoc(), DataConsumer);
          }
        }
      }
    }
  }
}

void index::indexASTUnit(ASTUnit &Unit, IndexDataConsumer &DataConsumer,
                         IndexingOptions Opts) {
  IndexingContext IndexCtx(Opts, DataConsumer);
  IndexCtx.setASTContext(Unit.getASTContext());
  DataConsumer.initialize(Unit.getASTContext());
  DataConsumer.setPreprocessor(Unit.getPreprocessorPtr());

  if (Opts.IndexMacrosInPreprocessor)
    indexPreprocessorMacros(Unit.getPreprocessor(), DataConsumer);
  indexTranslationUnit(Unit, IndexCtx);
  DataConsumer.finish();
}

void index::indexTopLevelDecls(ASTContext &Ctx, Preprocessor &PP,
                               ArrayRef<const Decl *> Decls,
                               IndexDataConsumer &DataConsumer,
                               IndexingOptions Opts) {
  IndexingContext IndexCtx(Opts, DataConsumer);
  IndexCtx.setASTContext(Ctx);

  DataConsumer.initialize(Ctx);

  if (Opts.IndexMacrosInPreprocessor)
    indexPreprocessorMacros(PP, DataConsumer);

  for (const Decl *D : Decls)
    IndexCtx.indexTopLevelDecl(D);
  DataConsumer.finish();
}

std::unique_ptr<PPCallbacks>
index::indexMacrosCallback(IndexDataConsumer &Consumer, IndexingOptions Opts) {
  return std::make_unique<IndexPPCallbacks>(
      std::make_shared<IndexingContext>(Opts, Consumer));
}

void index::indexModuleFile(serialization::ModuleFile &Mod, ASTReader &Reader,
                            IndexDataConsumer &DataConsumer,
                            IndexingOptions Opts) {
  ASTContext &Ctx = Reader.getContext();
  IndexingContext IndexCtx(Opts, DataConsumer);
  IndexCtx.setASTContext(Ctx);
  DataConsumer.initialize(Ctx);

  if (Opts.IndexMacrosInPreprocessor) {
    indexPreprocessorModuleMacros(Reader.getPreprocessor(), Mod, DataConsumer);
  }

  for (const Decl *D : Reader.getModuleFileLevelDecls(Mod)) {
    IndexCtx.indexTopLevelDecl(D);
  }
  DataConsumer.finish();
}
