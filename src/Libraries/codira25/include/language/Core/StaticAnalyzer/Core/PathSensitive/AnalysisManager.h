/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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

//== AnalysisManager.h - Path sensitive analysis data manager ------*- C++ -*-//
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
// This file defines the AnalysisManager class that manages the data and policy
// for path sensitive analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_CORE_STATICANALYZER_CORE_PATHSENSITIVE_ANALYSISMANAGER_H
#define LANGUAGE_CORE_STATICANALYZER_CORE_PATHSENSITIVE_ANALYSISMANAGER_H

#include "language/Core/Analysis/AnalysisDeclContext.h"
#include "language/Core/Analysis/PathDiagnostic.h"
#include "language/Core/Lex/Preprocessor.h"
#include "language/Core/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "language/Core/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "language/Core/StaticAnalyzer/Core/PathDiagnosticConsumers.h"

namespace language::Core {

class CodeInjector;

namespace ento {
  class CheckerManager;

class AnalysisManager : public BugReporterData {
  virtual void anchor();
  AnalysisDeclContextManager AnaCtxMgr;

  ASTContext &Ctx;
  Preprocessor &PP;
  const LangOptions &LangOpts;
  PathDiagnosticConsumers PathConsumers;

  // Configurable components creators.
  StoreManagerCreator CreateStoreMgr;
  ConstraintManagerCreator CreateConstraintMgr;

  CheckerManager *CheckerMgr;

public:
  AnalyzerOptions &options;

  AnalysisManager(ASTContext &ctx, Preprocessor &PP,
                  PathDiagnosticConsumers Consumers,
                  StoreManagerCreator storemgr,
                  ConstraintManagerCreator constraintmgr,
                  CheckerManager *checkerMgr, AnalyzerOptions &Options,
                  std::unique_ptr<CodeInjector> injector = nullptr);

  ~AnalysisManager() override;

  void ClearContexts() {
    AnaCtxMgr.clear();
  }

  AnalysisDeclContextManager& getAnalysisDeclContextManager() {
    return AnaCtxMgr;
  }

  Preprocessor &getPreprocessor() override { return PP; }

  StoreManagerCreator getStoreManagerCreator() {
    return CreateStoreMgr;
  }

  AnalyzerOptions& getAnalyzerOptions() override {
    return options;
  }

  ConstraintManagerCreator getConstraintManagerCreator() {
    return CreateConstraintMgr;
  }

  CheckerManager *getCheckerManager() const { return CheckerMgr; }

  ASTContext &getASTContext() override {
    return Ctx;
  }

  SourceManager &getSourceManager() override {
    return getASTContext().getSourceManager();
  }

  const LangOptions &getLangOpts() const {
    return LangOpts;
  }

  ArrayRef<std::unique_ptr<PathDiagnosticConsumer>>
  getPathDiagnosticConsumers() override {
    return PathConsumers;
  }

  void FlushDiagnostics();

  bool shouldVisualize() const {
    return options.visualizeExplodedGraphWithGraphViz;
  }

  bool shouldInlineCall() const {
    return options.getIPAMode() != IPAK_None;
  }

  CFG *getCFG(Decl const *D) {
    return AnaCtxMgr.getContext(D)->getCFG();
  }

  template <typename T>
  T *getAnalysis(Decl const *D) {
    return AnaCtxMgr.getContext(D)->getAnalysis<T>();
  }

  ParentMap &getParentMap(Decl const *D) {
    return AnaCtxMgr.getContext(D)->getParentMap();
  }

  AnalysisDeclContext *getAnalysisDeclContext(const Decl *D) {
    return AnaCtxMgr.getContext(D);
  }

  static bool isInCodeFile(SourceLocation SL, const SourceManager &SM) {
    if (SM.isInMainFile(SL))
      return true;

    // Support the "unified sources" compilation method (eg. WebKit) that
    // involves producing non-header files that include other non-header files.
    // We should be included directly from a UnifiedSource* file
    // and we shouldn't be a header - which is a very safe defensive check.
    SourceLocation IL = SM.getIncludeLoc(SM.getFileID(SL));
    if (!IL.isValid() || !SM.isInMainFile(IL))
      return false;
    // Should rather be "file name starts with", but the current .getFilename
    // includes the full path.
    if (SM.getFilename(IL).contains("UnifiedSource")) {
      // It might be great to reuse FrontendOptions::getInputKindForExtension()
      // but for now it doesn't discriminate between code and header files.
      return toolchain::StringSwitch<bool>(SM.getFilename(SL).rsplit('.').second)
          .Cases("c", "m", "mm", "C", "cc", "cp", true)
          .Cases("cpp", "CPP", "c++", "cxx", "cppm", true)
          .Default(false);
    }

    return false;
  }

  bool isInCodeFile(SourceLocation SL) {
    const SourceManager &SM = getASTContext().getSourceManager();
    return isInCodeFile(SL, SM);
  }
};

} // enAnaCtxMgrspace

} // end clang namespace

#endif
