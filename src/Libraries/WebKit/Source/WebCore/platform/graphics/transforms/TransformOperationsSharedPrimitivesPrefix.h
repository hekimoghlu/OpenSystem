/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
#include <optional>
#include <wtf/Vector.h>

namespace WebCore {

class TransformOperations;

// This class is used to find a shared prefix of transform function primitives (as
// defined by CSS Transforms Level 1 & 2). Given a series of `TransformOperations` in
// the keyframes of an animation. After `update()` is called with the `TransformOperations`
// of every keyframe, `primitives()` will return the prefix of primitives that are shared
// by all keyframes passed to `update()`.
class TransformOperationsSharedPrimitivesPrefix final {
public:
    void update(const TransformOperations&);

    bool hadIncompatibleTransformFunctions() { return m_indexOfFirstMismatch.has_value(); }
    const Vector<TransformOperation::Type>& primitives() const { return m_primitives; }

private:
    std::optional<size_t> m_indexOfFirstMismatch;
    Vector<TransformOperation::Type> m_primitives;
};

} // namespace WebCore

