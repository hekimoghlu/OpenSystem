/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 17, 2025.
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
#include "config.h"
#include "CachedRecovery.h"

#if ENABLE(JIT)

namespace JSC {

// We prefer loading doubles and undetermined JSValues into FPRs
// because it would otherwise use up GPRs.  Two in JSVALUE32_64.
bool CachedRecovery::loadsIntoFPR() const
{
    switch (recovery().technique()) {
    case DoubleDisplacedInJSStack:
    case DisplacedInJSStack:
#if USE(JSVALUE64)
    case CellDisplacedInJSStack:
#endif
        return true;

    default:
        return false;
    }
}

// Integers, booleans and cells can be loaded into GPRs
bool CachedRecovery::loadsIntoGPR() const
{
    switch (recovery().technique()) {
    case Int32DisplacedInJSStack:
#if USE(JSVALUE32_64)
    case Int32TagDisplacedInJSStack:
#elif USE(JSVALUE64)
    case Int52DisplacedInJSStack:
    case StrictInt52DisplacedInJSStack:
    case DisplacedInJSStack:
#endif
    case BooleanDisplacedInJSStack:
    case CellDisplacedInJSStack:
        return true;

    default:
        return false;
    }
}

} // namespace JSC

#endif // ENABLE(JIT)
