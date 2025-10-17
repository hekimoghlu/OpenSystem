/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 1, 2024.
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

#include "TransformOperation.h"
#include <wtf/Ref.h>

namespace WebCore {

struct BlendingContext;

class IdentityTransformOperation final : public TransformOperation {
public:
    WEBCORE_EXPORT static Ref<IdentityTransformOperation> create();

    Ref<TransformOperation> clone() const override
    {
        return create();
    }

private:
    bool isIdentity() const override { return true; }

    bool operator==(const TransformOperation& o) const override
    {
        return isSameType(o);
    }

    bool apply(TransformationMatrix&, const FloatSize&) const override
    {
        return false;
    }

    Ref<TransformOperation> blend(const TransformOperation*, const BlendingContext&, bool = false) override
    {
        return *this;
    }

    void dump(WTF::TextStream&) const final;

    IdentityTransformOperation();
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_TRANSFORMOPERATION(WebCore::IdentityTransformOperation, WebCore::TransformOperation::Type::Identity ==)
