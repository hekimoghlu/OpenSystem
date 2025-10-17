/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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

#include "DOMJITHeapRange.h"

namespace JSC { namespace DOMJIT {

struct Effect {

    constexpr static Effect forWrite(HeapRange writeRange)
    {
        return { HeapRange::none(), writeRange };
    }

    constexpr static Effect forRead(HeapRange readRange)
    {
        return { readRange, HeapRange::none() };
    }

    constexpr static Effect forReadWrite(HeapRange readRange, HeapRange writeRange)
    {
        return { readRange, writeRange };
    }

    constexpr static Effect forPure()
    {
        return { HeapRange::none(), HeapRange::none(), HeapRange::none() };
    }

    constexpr static Effect forDef(HeapRange def)
    {
        return { def, HeapRange::none(), def };
    }

    constexpr static Effect forDef(HeapRange def, HeapRange readRange, HeapRange writeRange)
    {
        return { readRange, writeRange, def };
    }

    constexpr bool mustGenerate() const
    {
        return !!writes;
    }

    HeapRange reads { HeapRange::top() };
    HeapRange writes { HeapRange::top() };
    HeapRange def { HeapRange::top() };
};

} }
