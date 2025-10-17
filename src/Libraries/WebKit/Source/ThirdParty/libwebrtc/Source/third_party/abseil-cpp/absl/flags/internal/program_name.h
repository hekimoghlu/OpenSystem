/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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

//
//  Copyright 2019 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ABSL_FLAGS_INTERNAL_PROGRAM_NAME_H_
#define ABSL_FLAGS_INTERNAL_PROGRAM_NAME_H_

#include <string>

#include "absl/base/config.h"
#include "absl/strings/string_view.h"

// --------------------------------------------------------------------
// Program name

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace flags_internal {

// Returns program invocation name or "UNKNOWN" if `SetProgramInvocationName()`
// is never called. At the moment this is always set to argv[0] as part of
// library initialization.
std::string ProgramInvocationName();

// Returns base name for program invocation name. For example, if
//   ProgramInvocationName() == "a/b/mybinary"
// then
//   ShortProgramInvocationName() == "mybinary"
std::string ShortProgramInvocationName();

// Sets program invocation name to a new value. Should only be called once
// during program initialization, before any threads are spawned.
void SetProgramInvocationName(absl::string_view prog_name_str);

}  // namespace flags_internal
ABSL_NAMESPACE_END
}  // namespace absl

#endif  // ABSL_FLAGS_INTERNAL_PROGRAM_NAME_H_
