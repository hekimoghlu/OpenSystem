/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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
#include "CanvasPattern.h"

#include "DOMMatrix2DInit.h"
#include "DOMMatrixReadOnly.h"
#include "NativeImage.h"
#include "Pattern.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

Ref<CanvasPattern> CanvasPattern::create(SourceImage&& image, bool repeatX, bool repeatY, bool originClean)
{
    return adoptRef(*new CanvasPattern(WTFMove(image), repeatX, repeatY, originClean));
}

CanvasPattern::CanvasPattern(SourceImage&& image, bool repeatX, bool repeatY, bool originClean)
    : m_pattern(Pattern::create(WTFMove(image), { repeatX, repeatY }))
    , m_originClean(originClean)
{
}

CanvasPattern::~CanvasPattern() = default;

bool CanvasPattern::parseRepetitionType(const String& type, bool& repeatX, bool& repeatY)
{
    if (type.isEmpty() || type == "repeat"_s) {
        repeatX = true;
        repeatY = true;
        return true;
    }
    if (type == "no-repeat"_s) {
        repeatX = false;
        repeatY = false;
        return true;
    }
    if (type == "repeat-x"_s) {
        repeatX = true;
        repeatY = false;
        return true;
    }
    if (type == "repeat-y"_s) {
        repeatX = false;
        repeatY = true;
        return true;
    }
    return false;
}

ExceptionOr<void> CanvasPattern::setTransform(DOMMatrix2DInit&& matrixInit)
{
    auto checkValid = DOMMatrixReadOnly::validateAndFixup(matrixInit);
    if (checkValid.hasException())
        return checkValid.releaseException();

    m_pattern->setPatternSpaceTransform({ matrixInit.a.value_or(1), matrixInit.b.value_or(0), matrixInit.c.value_or(0), matrixInit.d.value_or(1), matrixInit.e.value_or(0), matrixInit.f.value_or(0) });
    return { };
}

} // namespace WebCore
