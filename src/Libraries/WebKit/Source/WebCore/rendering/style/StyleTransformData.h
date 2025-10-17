/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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

#include "LengthPoint.h"
#include "RenderStyleConstants.h"
#include "TransformOperations.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleTransformData);
class StyleTransformData : public RefCounted<StyleTransformData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleTransformData);
public:
    static Ref<StyleTransformData> create() { return adoptRef(*new StyleTransformData); }
    Ref<StyleTransformData> copy() const;

    bool operator==(const StyleTransformData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleTransformData&) const;
#endif

    bool hasTransform() const { return operations.size(); }

    LengthPoint originXY() const { return { x, y }; }

    TransformOperations operations;
    Length x;
    Length y;
    float z;
    TransformBox transformBox;

private:
    StyleTransformData();
    StyleTransformData(const StyleTransformData&);
};

} // namespace WebCore
