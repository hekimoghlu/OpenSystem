/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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
#pragma once

#include <wtf/OptionSet.h>

namespace WGSL {

struct Type;
class TypeStore;


using Constraint = uint8_t;

namespace Constraints {

constexpr Constraint None = 0;

constexpr Constraint Bool          = 1 << 0;
constexpr Constraint AbstractInt   = 1 << 1;
constexpr Constraint I32           = 1 << 2;
constexpr Constraint U32           = 1 << 3;
constexpr Constraint AbstractFloat = 1 << 4;
constexpr Constraint F32           = 1 << 5;
constexpr Constraint F16           = 1 << 6;

constexpr Constraint ConcreteFloat = F16 | F32;
constexpr Constraint Float = ConcreteFloat | AbstractFloat;

constexpr Constraint ConcreteInteger = I32 | U32;
constexpr Constraint Integer = ConcreteInteger | AbstractInt;
constexpr Constraint SignedInteger = I32 | AbstractInt;

constexpr Constraint Scalar = Bool | Integer | Float;
constexpr Constraint ConcreteScalar = Bool | ConcreteInteger | ConcreteFloat;

constexpr Constraint Concrete32BitNumber = ConcreteInteger | F32;

constexpr Constraint SignedNumber = Float | SignedInteger;
constexpr Constraint Number = Float | Integer;

}

bool satisfies(const Type*, Constraint);
const Type* satisfyOrPromote(const Type*, Constraint, const TypeStore&);
const Type* concretize(const Type*, TypeStore&);

} // namespace WGSL
