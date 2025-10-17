/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
#include "AXImage.h"

#include "AXLogger.h"
#include "Chrome.h"
#include "ChromeClient.h"
#include "DocumentInlines.h"
#include "TextRecognitionOptions.h"

namespace WebCore {

AXImage::AXImage(AXID axID, RenderImage& renderer)
    : AccessibilityRenderObject(axID, renderer)
{
}

Ref<AXImage> AXImage::create(AXID axID, RenderImage& renderer)
{
    return adoptRef(*new AXImage(axID, renderer));
}

AccessibilityRole AXImage::determineAccessibilityRole()
{
    if ((m_ariaRole = determineAriaRoleAttribute()) != AccessibilityRole::Unknown)
        return m_ariaRole;
    return AccessibilityRole::Image;
}

std::optional<AXCoreObject::AccessibilityChildrenVector> AXImage::imageOverlayElements()
{
    AXTRACE("AXImage::imageOverlayElements"_s);

    const auto& children = this->unignoredChildren();
    if (children.size())
        return children;

#if ENABLE(IMAGE_ANALYSIS)
    RefPtr page = this->page();
    if (!page)
        return std::nullopt;

    RefPtr element = this->element();
    if (!element)
        return std::nullopt;

    page->chrome().client().requestTextRecognition(*element, { }, [] (RefPtr<Element>&& imageOverlayHost) {
        if (!imageOverlayHost)
            return;

        if (CheckedPtr axObjectCache = imageOverlayHost->document().existingAXObjectCache())
            axObjectCache->postNotification(imageOverlayHost.get(), AXNotification::ImageOverlayChanged);
    });
#endif

    return std::nullopt;
}

} // namespace WebCore
