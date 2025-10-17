/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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

// Copyright 2018 The Abseil Authors.
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

#include <cstdint>

#include "absl/base/internal/raw_logging.h"

namespace absl {
ABSL_NAMESPACE_BEGIN

namespace debugging_internal {

int InstallSymbolDecorator(SymbolDecorator, void*) { return -1; }
bool RemoveSymbolDecorator(int) { return false; }
bool RemoveAllSymbolDecorators(void) { return false; }
bool RegisterFileMappingHint(const void *, const void *, uint64_t, const char *) {
  return false;
}
bool GetFileMappingHint(const void **, const void **, uint64_t *, const char **) {
  return false;
}

}  // namespace debugging_internal

void InitializeSymbolizer(const char*) {}
bool Symbolize(const void *, char *, int) { return false; }

ABSL_NAMESPACE_END
}  // namespace absl
