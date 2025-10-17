/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 1, 2024.
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

//===--- CommentCommandTraits.cpp - Comment command properties --*- C++ -*-===//
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

#include "language/Core/AST/CommentCommandTraits.h"
#include <cassert>

namespace language::Core {
namespace comments {

#include "language/Core/AST/CommentCommandInfo.inc"

CommandTraits::CommandTraits(toolchain::BumpPtrAllocator &Allocator,
                             const CommentOptions &CommentOptions)
    : NextID(std::size(Commands)), Allocator(Allocator) {
  registerCommentOptions(CommentOptions);
}

void CommandTraits::registerCommentOptions(
    const CommentOptions &CommentOptions) {
  for (CommentOptions::BlockCommandNamesTy::const_iterator
           I = CommentOptions.BlockCommandNames.begin(),
           E = CommentOptions.BlockCommandNames.end();
       I != E; I++) {
    registerBlockCommand(*I);
  }
}

const CommandInfo *CommandTraits::getCommandInfoOrNULL(StringRef Name) const {
  if (const CommandInfo *Info = getBuiltinCommandInfo(Name))
    return Info;
  return getRegisteredCommandInfo(Name);
}

const CommandInfo *CommandTraits::getCommandInfo(unsigned CommandID) const {
  if (const CommandInfo *Info = getBuiltinCommandInfo(CommandID))
    return Info;
  return getRegisteredCommandInfo(CommandID);
}

const CommandInfo *
CommandTraits::getTypoCorrectCommandInfo(StringRef Typo) const {
  // Single-character command impostures, such as \t or \n, should not go
  // through the fixit logic.
  if (Typo.size() <= 1)
    return nullptr;

  // The maximum edit distance we're prepared to accept.
  const unsigned MaxEditDistance = 1;

  unsigned BestEditDistance = MaxEditDistance;
  SmallVector<const CommandInfo *, 2> BestCommand;

  auto ConsiderCorrection = [&](const CommandInfo *Command) {
    StringRef Name = Command->Name;

    unsigned MinPossibleEditDistance = abs((int)Name.size() - (int)Typo.size());
    if (MinPossibleEditDistance <= BestEditDistance) {
      unsigned EditDistance = Typo.edit_distance(Name, true, BestEditDistance);
      if (EditDistance < BestEditDistance) {
        BestEditDistance = EditDistance;
        BestCommand.clear();
      }
      if (EditDistance == BestEditDistance)
        BestCommand.push_back(Command);
    }
  };

  for (const auto &Command : Commands)
    ConsiderCorrection(&Command);

  for (const auto *Command : RegisteredCommands)
    if (!Command->IsUnknownCommand)
      ConsiderCorrection(Command);

  return BestCommand.size() == 1 ? BestCommand[0] : nullptr;
}

CommandInfo *CommandTraits::createCommandInfoWithName(StringRef CommandName) {
  char *Name = Allocator.Allocate<char>(CommandName.size() + 1);
  memcpy(Name, CommandName.data(), CommandName.size());
  Name[CommandName.size()] = '\0';

  // Value-initialize (=zero-initialize in this case) a new CommandInfo.
  CommandInfo *Info = new (Allocator) CommandInfo();
  Info->Name = Name;
  // We only have a limited number of bits to encode command IDs in the
  // CommandInfo structure, so the ID numbers can potentially wrap around.
  assert((NextID < (1 << CommandInfo::NumCommandIDBits))
         && "Too many commands. We have limited bits for the command ID.");
  Info->ID = NextID++;

  RegisteredCommands.push_back(Info);

  return Info;
}

const CommandInfo *CommandTraits::registerUnknownCommand(
                                                  StringRef CommandName) {
  CommandInfo *Info = createCommandInfoWithName(CommandName);
  Info->IsUnknownCommand = true;
  return Info;
}

const CommandInfo *CommandTraits::registerBlockCommand(StringRef CommandName) {
  CommandInfo *Info = createCommandInfoWithName(CommandName);
  Info->IsBlockCommand = true;
  return Info;
}

const CommandInfo *CommandTraits::getBuiltinCommandInfo(
                                                  unsigned CommandID) {
  if (CommandID < std::size(Commands))
    return &Commands[CommandID];
  return nullptr;
}

const CommandInfo *CommandTraits::getRegisteredCommandInfo(
                                                  StringRef Name) const {
  for (unsigned i = 0, e = RegisteredCommands.size(); i != e; ++i) {
    if (RegisteredCommands[i]->Name == Name)
      return RegisteredCommands[i];
  }
  return nullptr;
}

const CommandInfo *CommandTraits::getRegisteredCommandInfo(
                                                  unsigned CommandID) const {
  return RegisteredCommands[CommandID - std::size(Commands)];
}

} // end namespace comments
} // end namespace language::Core

