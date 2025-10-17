/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 28, 2024.
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

//===-- lib/Semantics/canonicalize-directives.cpp -------------------------===//
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

#include "canonicalize-directives.h"
#include "language/Compability/Parser/parse-tree-visitor.h"
#include "language/Compability/Semantics/tools.h"

namespace language::Compability::semantics {

using namespace parser::literals;

// Check that directives are associated with the correct constructs.
// Directives that need to be associated with other constructs in the execution
// part are moved to the execution part so they can be checked there.
class CanonicalizationOfDirectives {
public:
  CanonicalizationOfDirectives(parser::Messages &messages)
      : messages_{messages} {}

  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}

  // Move directives that must appear in the Execution part out of the
  // Specification part.
  void Post(parser::SpecificationPart &spec);
  bool Pre(parser::ExecutionPart &x);

  // Ensure that directives associated with constructs appear accompanying the
  // construct.
  void Post(parser::Block &block);

private:
  // Ensure that loop directives appear immediately before a loop.
  void CheckLoopDirective(parser::CompilerDirective &dir, parser::Block &block,
      std::list<parser::ExecutionPartConstruct>::iterator it);

  parser::Messages &messages_;

  // Directives to be moved to the Execution part from the Specification part.
  std::list<common::Indirection<parser::CompilerDirective>>
      directivesToConvert_;
};

bool CanonicalizeDirectives(
    parser::Messages &messages, parser::Program &program) {
  CanonicalizationOfDirectives dirs{messages};
  Walk(program, dirs);
  return !messages.AnyFatalError();
}

static bool IsExecutionDirective(const parser::CompilerDirective &dir) {
  return std::holds_alternative<parser::CompilerDirective::VectorAlways>(
             dir.u) ||
      std::holds_alternative<parser::CompilerDirective::Unroll>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::UnrollAndJam>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::NoVector>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::NoUnroll>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::NoUnrollAndJam>(dir.u);
}

void CanonicalizationOfDirectives::Post(parser::SpecificationPart &spec) {
  auto &list{
      std::get<std::list<common::Indirection<parser::CompilerDirective>>>(
          spec.t)};
  for (auto it{list.begin()}; it != list.end();) {
    if (IsExecutionDirective(it->value())) {
      directivesToConvert_.emplace_back(std::move(*it));
      it = list.erase(it);
    } else {
      ++it;
    }
  }
}

bool CanonicalizationOfDirectives::Pre(parser::ExecutionPart &x) {
  auto origFirst{x.v.begin()};
  for (auto &dir : directivesToConvert_) {
    x.v.insert(origFirst,
        parser::ExecutionPartConstruct{
            parser::ExecutableConstruct{std::move(dir)}});
  }

  directivesToConvert_.clear();
  return true;
}

void CanonicalizationOfDirectives::CheckLoopDirective(
    parser::CompilerDirective &dir, parser::Block &block,
    std::list<parser::ExecutionPartConstruct>::iterator it) {

  // Skip over this and other compiler directives
  while (it != block.end() && parser::Unwrap<parser::CompilerDirective>(*it)) {
    ++it;
  }

  if (it == block.end() ||
      (!parser::Unwrap<parser::DoConstruct>(*it) &&
          !parser::Unwrap<parser::OpenACCLoopConstruct>(*it) &&
          !parser::Unwrap<parser::OpenACCCombinedConstruct>(*it))) {
    std::string s{parser::ToUpperCaseLetters(dir.source.ToString())};
    s.pop_back(); // Remove trailing newline from source string
    messages_.Say(
        dir.source, "A DO loop must follow the %s directive"_warn_en_US, s);
  }
}

void CanonicalizationOfDirectives::Post(parser::Block &block) {
  for (auto it{block.begin()}; it != block.end(); ++it) {
    if (auto *dir{parser::Unwrap<parser::CompilerDirective>(*it)}) {
      std::visit(
          common::visitors{[&](parser::CompilerDirective::VectorAlways &) {
                             CheckLoopDirective(*dir, block, it);
                           },
              [&](parser::CompilerDirective::Unroll &) {
                CheckLoopDirective(*dir, block, it);
              },
              [&](parser::CompilerDirective::UnrollAndJam &) {
                CheckLoopDirective(*dir, block, it);
              },
              [&](parser::CompilerDirective::NoVector &) {
                CheckLoopDirective(*dir, block, it);
              },
              [&](parser::CompilerDirective::NoUnroll &) {
                CheckLoopDirective(*dir, block, it);
              },
              [&](parser::CompilerDirective::NoUnrollAndJam &) {
                CheckLoopDirective(*dir, block, it);
              },
              [&](auto &) {}},
          dir->u);
    }
  }
}

} // namespace language::Compability::semantics
