/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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

//===--- RefactoringCallbacks.cpp - Structural query framework ------------===//
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
//
//===----------------------------------------------------------------------===//
#include "language/Core/Tooling/RefactoringCallbacks.h"
#include "language/Core/ASTMatchers/ASTMatchFinder.h"
#include "language/Core/Basic/SourceLocation.h"
#include "language/Core/Lex/Lexer.h"

using toolchain::StringError;
using toolchain::make_error;

namespace language::Core {
namespace tooling {

RefactoringCallback::RefactoringCallback() {}
tooling::Replacements &RefactoringCallback::getReplacements() {
  return Replace;
}

ASTMatchRefactorer::ASTMatchRefactorer(
    std::map<std::string, Replacements> &FileToReplaces)
    : FileToReplaces(FileToReplaces) {}

void ASTMatchRefactorer::addDynamicMatcher(
    const ast_matchers::internal::DynTypedMatcher &Matcher,
    RefactoringCallback *Callback) {
  MatchFinder.addDynamicMatcher(Matcher, Callback);
  Callbacks.push_back(Callback);
}

class RefactoringASTConsumer : public ASTConsumer {
public:
  explicit RefactoringASTConsumer(ASTMatchRefactorer &Refactoring)
      : Refactoring(Refactoring) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    // The ASTMatchRefactorer is re-used between translation units.
    // Clear the matchers so that each Replacement is only emitted once.
    for (const auto &Callback : Refactoring.Callbacks) {
      Callback->getReplacements().clear();
    }
    Refactoring.MatchFinder.matchAST(Context);
    for (const auto &Callback : Refactoring.Callbacks) {
      for (const auto &Replacement : Callback->getReplacements()) {
        toolchain::Error Err =
            Refactoring.FileToReplaces[std::string(Replacement.getFilePath())]
                .add(Replacement);
        if (Err) {
          toolchain::errs() << "Skipping replacement " << Replacement.toString()
                       << " due to this error:\n"
                       << toString(std::move(Err)) << "\n";
        }
      }
    }
  }

private:
  ASTMatchRefactorer &Refactoring;
};

std::unique_ptr<ASTConsumer> ASTMatchRefactorer::newASTConsumer() {
  return std::make_unique<RefactoringASTConsumer>(*this);
}

static Replacement replaceStmtWithText(SourceManager &Sources, const Stmt &From,
                                       StringRef Text) {
  return tooling::Replacement(
      Sources, CharSourceRange::getTokenRange(From.getSourceRange()), Text);
}
static Replacement replaceStmtWithStmt(SourceManager &Sources, const Stmt &From,
                                       const Stmt &To) {
  return replaceStmtWithText(
      Sources, From,
      Lexer::getSourceText(CharSourceRange::getTokenRange(To.getSourceRange()),
                           Sources, LangOptions()));
}

ReplaceStmtWithText::ReplaceStmtWithText(StringRef FromId, StringRef ToText)
    : FromId(std::string(FromId)), ToText(std::string(ToText)) {}

void ReplaceStmtWithText::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const Stmt *FromMatch = Result.Nodes.getNodeAs<Stmt>(FromId)) {
    auto Err = Replace.add(tooling::Replacement(
        *Result.SourceManager,
        CharSourceRange::getTokenRange(FromMatch->getSourceRange()), ToText));
    // FIXME: better error handling. For now, just print error message in the
    // release version.
    if (Err) {
      toolchain::errs() << toolchain::toString(std::move(Err)) << "\n";
      assert(false);
    }
  }
}

ReplaceStmtWithStmt::ReplaceStmtWithStmt(StringRef FromId, StringRef ToId)
    : FromId(std::string(FromId)), ToId(std::string(ToId)) {}

void ReplaceStmtWithStmt::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const Stmt *FromMatch = Result.Nodes.getNodeAs<Stmt>(FromId);
  const Stmt *ToMatch = Result.Nodes.getNodeAs<Stmt>(ToId);
  if (FromMatch && ToMatch) {
    auto Err = Replace.add(
        replaceStmtWithStmt(*Result.SourceManager, *FromMatch, *ToMatch));
    // FIXME: better error handling. For now, just print error message in the
    // release version.
    if (Err) {
      toolchain::errs() << toolchain::toString(std::move(Err)) << "\n";
      assert(false);
    }
  }
}

ReplaceIfStmtWithItsBody::ReplaceIfStmtWithItsBody(StringRef Id,
                                                   bool PickTrueBranch)
    : Id(std::string(Id)), PickTrueBranch(PickTrueBranch) {}

void ReplaceIfStmtWithItsBody::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const IfStmt *Node = Result.Nodes.getNodeAs<IfStmt>(Id)) {
    const Stmt *Body = PickTrueBranch ? Node->getThen() : Node->getElse();
    if (Body) {
      auto Err =
          Replace.add(replaceStmtWithStmt(*Result.SourceManager, *Node, *Body));
      // FIXME: better error handling. For now, just print error message in the
      // release version.
      if (Err) {
        toolchain::errs() << toolchain::toString(std::move(Err)) << "\n";
        assert(false);
      }
    } else if (!PickTrueBranch) {
      // If we want to use the 'else'-branch, but it doesn't exist, delete
      // the whole 'if'.
      auto Err =
          Replace.add(replaceStmtWithText(*Result.SourceManager, *Node, ""));
      // FIXME: better error handling. For now, just print error message in the
      // release version.
      if (Err) {
        toolchain::errs() << toolchain::toString(std::move(Err)) << "\n";
        assert(false);
      }
    }
  }
}

ReplaceNodeWithTemplate::ReplaceNodeWithTemplate(
    toolchain::StringRef FromId, std::vector<TemplateElement> Template)
    : FromId(std::string(FromId)), Template(std::move(Template)) {}

toolchain::Expected<std::unique_ptr<ReplaceNodeWithTemplate>>
ReplaceNodeWithTemplate::create(StringRef FromId, StringRef ToTemplate) {
  std::vector<TemplateElement> ParsedTemplate;
  for (size_t Index = 0; Index < ToTemplate.size();) {
    if (ToTemplate[Index] == '$') {
      if (ToTemplate.substr(Index, 2) == "$$") {
        Index += 2;
        ParsedTemplate.push_back(
            TemplateElement{TemplateElement::Literal, "$"});
      } else if (ToTemplate.substr(Index, 2) == "${") {
        size_t EndOfIdentifier = ToTemplate.find("}", Index);
        if (EndOfIdentifier == std::string::npos) {
          return make_error<StringError>(
              "Unterminated ${...} in replacement template near " +
                  ToTemplate.substr(Index),
              toolchain::inconvertibleErrorCode());
        }
        std::string SourceNodeName = std::string(
            ToTemplate.substr(Index + 2, EndOfIdentifier - Index - 2));
        ParsedTemplate.push_back(
            TemplateElement{TemplateElement::Identifier, SourceNodeName});
        Index = EndOfIdentifier + 1;
      } else {
        return make_error<StringError>(
            "Invalid $ in replacement template near " +
                ToTemplate.substr(Index),
            toolchain::inconvertibleErrorCode());
      }
    } else {
      size_t NextIndex = ToTemplate.find('$', Index + 1);
      ParsedTemplate.push_back(TemplateElement{
          TemplateElement::Literal,
          std::string(ToTemplate.substr(Index, NextIndex - Index))});
      Index = NextIndex;
    }
  }
  return std::unique_ptr<ReplaceNodeWithTemplate>(
      new ReplaceNodeWithTemplate(FromId, std::move(ParsedTemplate)));
}

void ReplaceNodeWithTemplate::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto &NodeMap = Result.Nodes.getMap();

  std::string ToText;
  for (const auto &Element : Template) {
    switch (Element.Type) {
    case TemplateElement::Literal:
      ToText += Element.Value;
      break;
    case TemplateElement::Identifier: {
      auto NodeIter = NodeMap.find(Element.Value);
      if (NodeIter == NodeMap.end()) {
        toolchain::errs() << "Node " << Element.Value
                     << " used in replacement template not bound in Matcher \n";
        toolchain::report_fatal_error("Unbound node in replacement template.");
      }
      CharSourceRange Source =
          CharSourceRange::getTokenRange(NodeIter->second.getSourceRange());
      ToText += Lexer::getSourceText(Source, *Result.SourceManager,
                                     Result.Context->getLangOpts());
      break;
    }
    }
  }
  auto It = NodeMap.find(FromId);
  if (It == NodeMap.end()) {
    toolchain::errs() << "Node to be replaced " << FromId
                 << " not bound in query.\n";
    toolchain::report_fatal_error("FromId node not bound in MatchResult");
  }
  auto Replacement =
      tooling::Replacement(*Result.SourceManager, &It->second, ToText,
                           Result.Context->getLangOpts());
  toolchain::Error Err = Replace.add(Replacement);
  if (Err) {
    toolchain::errs() << "Query and replace failed in " << Replacement.getFilePath()
                 << "! " << toolchain::toString(std::move(Err)) << "\n";
    toolchain::report_fatal_error("Replacement failed");
  }
}

} // end namespace tooling
} // end namespace language::Core
