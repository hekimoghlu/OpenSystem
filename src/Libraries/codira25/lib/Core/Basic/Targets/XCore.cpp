/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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

//===--- XCore.cpp - Implement XCore target feature support ---------------===//
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
// This file implements XCore TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "XCore.h"
#include "language/Core/Basic/Builtins.h"
#include "language/Core/Basic/MacroBuilder.h"
#include "language/Core/Basic/TargetBuiltins.h"

using namespace language::Core;
using namespace language::Core::targets;

static constexpr int NumBuiltins =
    XCore::LastTSBuiltin - Builtin::FirstTSBuiltin;

static constexpr toolchain::StringTable BuiltinStrings =
    CLANG_BUILTIN_STR_TABLE_START
#define BUILTIN CLANG_BUILTIN_STR_TABLE
#include "language/Core/Basic/BuiltinsXCore.def"
    ;

static constexpr auto BuiltinInfos = Builtin::MakeInfos<NumBuiltins>({
#define BUILTIN CLANG_BUILTIN_ENTRY
#define LIBBUILTIN CLANG_LIBBUILTIN_ENTRY
#include "language/Core/Basic/BuiltinsXCore.def"
});

void XCoreTargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  Builder.defineMacro("__xcore__");
  Builder.defineMacro("__XS1B__");
}

toolchain::SmallVector<Builtin::InfosShard>
XCoreTargetInfo::getTargetBuiltins() const {
  return {{&BuiltinStrings, BuiltinInfos}};
}
