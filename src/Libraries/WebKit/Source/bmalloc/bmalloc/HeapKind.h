/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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

#include "BAssert.h"
#include "BInline.h"
#include "Gigacage.h"

namespace bmalloc {

enum class HeapKind {
    Primary,
    PrimitiveGigacage,
};

static constexpr unsigned numHeaps = 2;

BINLINE bool isGigacage(HeapKind heapKind)
{
    switch (heapKind) {
    case HeapKind::Primary:
        return false;
    case HeapKind::PrimitiveGigacage:
        return true;
    }
    BCRASH();
    return false;
}

BINLINE Gigacage::Kind gigacageKind(HeapKind kind)
{
    switch (kind) {
    case HeapKind::Primary:
        BCRASH();
        return Gigacage::Primitive;
    case HeapKind::PrimitiveGigacage:
        return Gigacage::Primitive;
    }
    BCRASH();
    return Gigacage::Primitive;
}

BINLINE HeapKind heapKind(Gigacage::Kind kind)
{
    switch (kind) {
    case Gigacage::Primitive:
        return HeapKind::PrimitiveGigacage;
    case Gigacage::NumberOfKinds:
        break;
    }
    BCRASH();
    return HeapKind::Primary;
}

BINLINE bool isActiveHeapKindAfterEnsuringGigacage(HeapKind kind)
{
    switch (kind) {
    case HeapKind::PrimitiveGigacage:
        if (Gigacage::isEnabled())
            return true;
        return false;
    default:
        return true;
    }
}

BEXPORT bool isActiveHeapKind(HeapKind);

BINLINE HeapKind mapToActiveHeapKindAfterEnsuringGigacage(HeapKind kind)
{
    switch (kind) {
    case HeapKind::PrimitiveGigacage:
        if (Gigacage::isEnabled())
            return kind;
        return HeapKind::Primary;
    default:
        return kind;
    }
}

BEXPORT HeapKind mapToActiveHeapKind(HeapKind);

} // namespace bmalloc

