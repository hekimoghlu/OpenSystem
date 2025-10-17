/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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

#include "DOMMatrix.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

enum class CSSTransformType : uint8_t {
    MatrixComponent,
    Perspective,
    Rotate,
    Scale,
    Skew,
    SkewX,
    SkewY,
    Translate
};

class DOMMatrix;
template<typename> class ExceptionOr;

class CSSTransformComponent : public RefCounted<CSSTransformComponent> {
protected:
    enum class Is2D : bool { No, Yes };
    CSSTransformComponent(Is2D is2D)
        : m_is2D(is2D) { }
public:
    String toString() const;
    virtual void serialize(StringBuilder&) const = 0;
    bool is2D() const { return m_is2D == Is2D::Yes; }
    virtual void setIs2D(bool is2D) { m_is2D = is2D ? Is2D::Yes : Is2D::No; }
    virtual ExceptionOr<Ref<DOMMatrix>> toMatrix() = 0;
    virtual ~CSSTransformComponent() = default;
    virtual CSSTransformType getType() const = 0;

    virtual RefPtr<CSSValue> toCSSValue() const = 0;

private:
    Is2D m_is2D;
};

} // namespace WebCore
