/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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

#include "language/ABI/Metadata.h"
#include "language/Demangling/Demangle.h"
#include "language/RemoteInspection/TypeRefBuilder.h"
#include "language/Remote/MetadataReader.h"
#include "language/Remote/InProcessMemoryReader.h"

#include "Private.h"

#include <cstdio>

using namespace language;

static std::string nameForMetadata(const Metadata *md)
{
  Demangle::__runtime::StackAllocatedDemangler<1024> dem;
  auto nodeTree = _language_buildDemanglingForMetadata(md, dem);
  if (!nodeTree)
    return "<unknown>";

  std::string result = Demangle::__runtime::nodeToString(nodeTree);
  return result;
}

extern "C" LANGUAGE_CC(language) void roundTripType(const Metadata *md) {
  // Get a name for it
  const std::string mdName = ::nameForMetadata(md);

  // Convert it to a Node tree
  Demangle::__runtime::StackAllocatedDemangler<1024> dem;
  auto nodeTree = _language_buildDemanglingForMetadata(md, dem);

  // Mangle that
  auto mangling = Demangle::__runtime::mangleNode(nodeTree);
  if (!mangling.isSuccess()) {
    printf("FAIL: %s (%p) -> mangling error %d:%u\n", mdName.c_str(), md,
           mangling.error().code, mangling.error().line);
    return;
  }
  std::string mangledName = mangling.result();

  // Look up the result
  auto result = language_getTypeByMangledName(MetadataState::Abstract,
                              mangledName,
                              nullptr,
                              [](unsigned, unsigned){ return nullptr; },
                              [](const Metadata *, unsigned) { return nullptr; });
  if (result.isError()) {
    auto err = result.getError();
    char *errStr = err->copyErrorString();
    printf("FAIL: %s (%p) -> %s -> ERROR %s\n",
           mdName.c_str(), md, mangledName.c_str(), errStr);
    err->freeErrorString(errStr);
    nodeTree->dump();

    result = language_getTypeByMangledNode(MetadataState::Abstract,
                                        dem,
                                        nodeTree,
                                        nullptr,
                                        [](unsigned, unsigned){ return nullptr; },
                                        [](const Metadata *, unsigned) { return nullptr; });
    if (result.isError()) {
      err = result.getError();
      char *errStr = err->copyErrorString();
      printf("=> Also failed on node: %s\n", errStr);
      err->freeErrorString(errStr);
    }
    return;
  }

  const Metadata *md2 = result.getType().getMetadata();

  std::string md2Name = "<FAIL>";

  if (md2)
    md2Name = ::nameForMetadata(md2);

  printf("%s: %s (%p) -> %s -> %s (%p)\n",
         md == md2 ? "PASS" : "FAIL",
         mdName.c_str(), md, mangledName.c_str(), md2Name.c_str(), md2);
}
