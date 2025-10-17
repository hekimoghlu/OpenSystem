/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
#include "PathOperation.h"

#include "AnimationUtilities.h"
#include "CSSRayValue.h"
#include "SVGElement.h"
#include "SVGElementTypeHelpers.h"
#include "SVGPathData.h"
#include "SVGPathElement.h"
#include "StylePrimitiveNumericTypes+Blending.h"
#include "StylePrimitiveNumericTypes+Conversions.h"

namespace WebCore {

PathOperation::~PathOperation() = default;

Ref<ReferencePathOperation> ReferencePathOperation::create(const String& url, const AtomString& fragment, const RefPtr<SVGElement> element)
{
    return adoptRef(*new ReferencePathOperation(url, fragment, element));
}

Ref<ReferencePathOperation> ReferencePathOperation::create(std::optional<Path>&& path)
{
    return adoptRef(*new ReferencePathOperation(WTFMove(path)));
}

Ref<PathOperation> ReferencePathOperation::clone() const
{
    if (auto path = this->path()) {
        auto pathCopy = *path;
        return adoptRef(*new ReferencePathOperation(WTFMove(pathCopy)));
    }
    return adoptRef(*new ReferencePathOperation(std::nullopt));
}

ReferencePathOperation::ReferencePathOperation(const String& url, const AtomString& fragment, const RefPtr<SVGElement> element)
    : PathOperation(Type::Reference)
    , m_url(url)
    , m_fragment(fragment)
{
    if (is<SVGPathElement>(element) || is<SVGGeometryElement>(element))
        m_path = pathFromGraphicsElement(*element);
}

ReferencePathOperation::ReferencePathOperation(std::optional<Path>&& path)
    : PathOperation(Type::Reference)
    , m_path(WTFMove(path))
{
}

// MARK: - ShapePathOperation

Ref<ShapePathOperation> ShapePathOperation::create(Style::BasicShape shape, CSSBoxType referenceBox)
{
    return adoptRef(*new ShapePathOperation(WTFMove(shape), referenceBox));
}

Ref<PathOperation> ShapePathOperation::clone() const
{
    return adoptRef(*new ShapePathOperation(m_shape, m_referenceBox));
}

bool ShapePathOperation::canBlend(const PathOperation& to) const
{
    RefPtr toOperation = dynamicDowncast<ShapePathOperation>(to);
    return toOperation && WebCore::Style::canBlend(m_shape, toOperation->m_shape);
}

RefPtr<PathOperation> ShapePathOperation::blend(const PathOperation* to, const BlendingContext& context) const
{
    Ref toShapePathOperation = downcast<ShapePathOperation>(*to);
    return ShapePathOperation::create(WebCore::Style::blend(m_shape, toShapePathOperation->m_shape, context));
}

std::optional<Path> ShapePathOperation::getPath(const TransformOperationData& data) const
{
    return MotionPath::computePathForShape(*this, data);
}

// MARK: - BoxPathOperation

Ref<BoxPathOperation> BoxPathOperation::create(CSSBoxType referenceBox)
{
    return adoptRef(*new BoxPathOperation(referenceBox));
}

Ref<PathOperation> BoxPathOperation::clone() const
{
    return adoptRef(*new BoxPathOperation(referenceBox()));
}

std::optional<Path> BoxPathOperation::getPath(const TransformOperationData& data) const
{
    return MotionPath::computePathForBox(*this, data);
}

// MARK: - RayPathOperation

Ref<RayPathOperation> RayPathOperation::create(Style::RayFunction ray, CSSBoxType referenceBox)
{
    return adoptRef(*new RayPathOperation(WTFMove(ray), referenceBox));
}

Ref<PathOperation> RayPathOperation::clone() const
{
    return adoptRef(*new RayPathOperation(m_ray, m_referenceBox));
}

bool RayPathOperation::canBlend(const PathOperation& to) const
{
    RefPtr toRayPathOperation = dynamicDowncast<RayPathOperation>(to);
    return toRayPathOperation && Style::canBlend(m_ray, toRayPathOperation->m_ray) && m_referenceBox == toRayPathOperation->referenceBox();
}

RefPtr<PathOperation> RayPathOperation::blend(const PathOperation* to, const BlendingContext& context) const
{
    Ref toRayPathOperation = downcast<RayPathOperation>(*to);
    return RayPathOperation::create(Style::blend(m_ray, toRayPathOperation->m_ray, context), m_referenceBox);
}

std::optional<Path> RayPathOperation::getPath(const TransformOperationData& data) const
{
    return MotionPath::computePathForRay(*this, data);
}

} // namespace WebCore
