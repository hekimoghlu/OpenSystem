/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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

#include "CPU.h"
#include "Options.h"

namespace JSC {

inline bool optimizeForARMv7IDIVSupported()
{
    return isARMv7IDIVSupported() && Options::useArchitectureSpecificOptimizations();
}

inline bool optimizeForARM64()
{
    return isARM64() && Options::useArchitectureSpecificOptimizations();
}

inline bool optimizeForX86()
{
    return isX86() && Options::useArchitectureSpecificOptimizations();
}

inline bool optimizeForX86_64()
{
    return isX86_64() && Options::useArchitectureSpecificOptimizations();
}

inline bool hasSensibleDoubleToInt()
{
    return optimizeForX86();
}

} // namespace JSC
