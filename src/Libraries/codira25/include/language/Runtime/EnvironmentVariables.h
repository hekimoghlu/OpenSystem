/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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

//===--- EnvironmentVariables.h - Debug variables. --------------*- C++ -*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//
//
// Debug behavior conditionally enabled using environment variables.
//
//===----------------------------------------------------------------------===//

#include "language/Threading/Once.h"
#include "language/shims/Visibility.h"

namespace language {
namespace runtime {
namespace environment {

void initialize(void *);

extern language::once_t initializeToken;

// Define a typedef "string" in language::runtime::environment to make string
// environment variables work
using string = const char *;

// Declare backing variables.
#define VARIABLE(name, type, defaultValue, help)                               \
  extern type name##_variable;                                                 \
  extern bool name##_isSet_variable;
#include "../../../stdlib/public/runtime/EnvironmentVariables.def"

// Define getter functions. This creates one function with the same name as the
// variable which returns the value set for that variable, and second function
// ending in _isSet which returns a boolean indicating whether the variable was
// set at all, to allow detecting when the variable was explicitly set to the
// same value as the default.
#define VARIABLE(name, type, defaultValue, help)                               \
  inline type name() {                                                         \
    language::once(initializeToken, initialize, nullptr);                         \
    return name##_variable;                                                    \
  }                                                                            \
  inline bool name##_isSet() {                                                 \
    language::once(initializeToken, initialize, nullptr);                         \
    return name##_isSet_variable;                                              \
  }
#include "../../../stdlib/public/runtime/EnvironmentVariables.def"

// Wrapper around LANGUAGE_DEBUG_CONCURRENCY_ENABLE_COOPERATIVE_QUEUES that the
// Concurrency library can call.
LANGUAGE_RUNTIME_STDLIB_SPI bool concurrencyEnableCooperativeQueues();

// Wrapper around LANGUAGE_DEBUG_VALIDATE_UNCHECKED_CONTINUATIONS that the
// Concurrency library can call.
LANGUAGE_RUNTIME_STDLIB_SPI bool concurrencyValidateUncheckedContinuations();

// Wrapper around LANGUAGE_IS_CURRENT_EXECUTOR_LEGACY_MODE_OVERRIDE that the
// Concurrency library can call.
LANGUAGE_RUNTIME_STDLIB_SPI const char *concurrencyIsCurrentExecutorLegacyModeOverride();

} // end namespace environment
} // end namespace runtime
} // end namespace language
