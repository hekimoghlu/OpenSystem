/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#include "PlatformCAAnimation.h"

#include <wtf/text/MakeString.h>
#include <wtf/text/StringToIntegerConversion.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

TextStream& operator<<(TextStream& ts, PlatformCAAnimation::AnimationType type)
{
    switch (type) {
    case PlatformCAAnimation::AnimationType::Basic: ts << "basic"; break;
    case PlatformCAAnimation::AnimationType::Group: ts << "group"; break;
    case PlatformCAAnimation::AnimationType::Keyframe: ts << "keyframe"; break;
    case PlatformCAAnimation::AnimationType::Spring: ts << "spring"; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, PlatformCAAnimation::FillModeType fillMode)
{
    switch (fillMode) {
    case PlatformCAAnimation::FillModeType::NoFillMode: ts << "none"; break;
    case PlatformCAAnimation::FillModeType::Forwards: ts << "forwards"; break;
    case PlatformCAAnimation::FillModeType::Backwards: ts << "backwards"; break;
    case PlatformCAAnimation::FillModeType::Both: ts << "both"; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, PlatformCAAnimation::ValueFunctionType valueFunctionType)
{
    switch (valueFunctionType) {
    case PlatformCAAnimation::ValueFunctionType::NoValueFunction: ts << "none"; break;
    case PlatformCAAnimation::ValueFunctionType::RotateX: ts << "rotateX"; break;
    case PlatformCAAnimation::ValueFunctionType::RotateY: ts << "rotateY"; break;
    case PlatformCAAnimation::ValueFunctionType::RotateZ: ts << "rotateZ"; break;
    case PlatformCAAnimation::ValueFunctionType::ScaleX: ts << "scaleX"; break;
    case PlatformCAAnimation::ValueFunctionType::ScaleY: ts << "scaleY"; break;
    case PlatformCAAnimation::ValueFunctionType::ScaleZ: ts << "scaleZ"; break;
    case PlatformCAAnimation::ValueFunctionType::Scale: ts << "scale"; break;
    case PlatformCAAnimation::ValueFunctionType::TranslateX: ts << "translateX"; break;
    case PlatformCAAnimation::ValueFunctionType::TranslateY: ts << "translateY"; break;
    case PlatformCAAnimation::ValueFunctionType::TranslateZ: ts << "translateZ"; break;
    case PlatformCAAnimation::ValueFunctionType::Translate: ts << "translate"; break;
    }
    return ts;
}

bool PlatformCAAnimation::isBasicAnimation() const
{
    return animationType() == AnimationType::Basic || animationType() == AnimationType::Spring;
}

static constexpr auto transformKeyPath = "transform"_s;
static constexpr auto opacityKeyPath = "opacity"_s;
static constexpr auto backgroundColorKeyPath = "backgroundColor"_s;
static constexpr auto filterKeyPathPrefix = "filters.filter_"_s;
static constexpr auto backdropFiltersKeyPath = "backdropFilters"_s;

String PlatformCAAnimation::makeGroupKeyPath()
{
    return emptyString();
}

String PlatformCAAnimation::makeKeyPath(AnimatedProperty animatedProperty, FilterOperation::Type filterOperationType, int index)
{
    switch (animatedProperty) {
    case AnimatedProperty::Translate:
    case AnimatedProperty::Scale:
    case AnimatedProperty::Rotate:
    case AnimatedProperty::Transform:
        return transformKeyPath;
    case AnimatedProperty::Opacity:
        return opacityKeyPath;
    case AnimatedProperty::BackgroundColor:
        return backgroundColorKeyPath;
    case AnimatedProperty::Filter:
        return makeString(filterKeyPathPrefix, index, '.', PlatformCAFilters::animatedFilterPropertyName(filterOperationType));
    case AnimatedProperty::WebkitBackdropFilter:
        return backdropFiltersKeyPath;
    case AnimatedProperty::Invalid:
        ASSERT_NOT_REACHED();
        return emptyString();
    }
    ASSERT_NOT_REACHED();
    return emptyString();
}

static bool isValidFilterKeyPath(const String& keyPath)
{
    if (!keyPath.startsWith(filterKeyPathPrefix))
        return false;

    size_t underscoreIndex = filterKeyPathPrefix.length();
    auto dotIndex = keyPath.find('.', underscoreIndex);
    if (dotIndex == notFound || dotIndex <= underscoreIndex)
        return false;

    auto indexString = keyPath.substring(underscoreIndex, dotIndex - underscoreIndex);
    auto parsedIndex = parseInteger<unsigned>(indexString);
    if (!parsedIndex)
        return false;

    auto filterOperationTypeString = keyPath.substring(dotIndex + 1);
    return PlatformCAFilters::isValidAnimatedFilterPropertyName(filterOperationTypeString);
}

bool PlatformCAAnimation::isValidKeyPath(const String& keyPath, AnimationType type)
{
    if (type == AnimationType::Group)
        return keyPath.isEmpty();

    if (keyPath == transformKeyPath
        || keyPath == opacityKeyPath
        || keyPath == backgroundColorKeyPath)
        return true;

    if (keyPath == backdropFiltersKeyPath)
        return true;

    if (isValidFilterKeyPath(keyPath))
        return true;

    return false;
}

} // namespace WebCore
