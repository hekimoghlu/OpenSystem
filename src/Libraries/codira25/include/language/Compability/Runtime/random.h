/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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

//===-- language/Compability/Runtime/random.h --------------------------*- C++ -*-===//
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

// Intrinsic subroutines RANDOM_INIT, RANDOM_NUMBER, and RANDOM_SEED.

#ifndef LANGUAGE_COMPABILITY_RUNTIME_RANDOM_H_
#define LANGUAGE_COMPABILITY_RUNTIME_RANDOM_H_

#include "language/Compability/Runtime/entry-names.h"
#include <cstdint>

namespace language::Compability::runtime {
class Descriptor;
extern "C" {

void RTNAME(RandomInit)(bool repeatable, bool image_distinct);

void RTNAME(RandomNumber)(
    const Descriptor &harvest, const char *source, int line);

// RANDOM_SEED may be called with a value for at most one of its three
// optional arguments.  Most calls map to an entry point for that value,
// or the entry point for no values.  If argument presence cannot be
// determined at compile time, function RandomSeed can be called to make
// the selection at run time.
void RTNAME(RandomSeedSize)(
    const Descriptor *size, const char *source, int line);
void RTNAME(RandomSeedPut)(const Descriptor *put, const char *source, int line);
void RTNAME(RandomSeedGet)(const Descriptor *get, const char *source, int line);
void RTNAME(RandomSeedDefaultPut)();
void RTNAME(RandomSeed)(const Descriptor *size, const Descriptor *put,
    const Descriptor *get, const char *source, int line);

} // extern "C"
} // namespace language::Compability::runtime

#endif // FORTRAN_RUNTIME_RANDOM_H_
