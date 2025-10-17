/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#include "StylePendingResources.h"

#include "CSSCursorImageValue.h"
#include "CachedResourceLoader.h"
#include "ContentData.h"
#include "CursorData.h"
#include "CursorList.h"
#include "DocumentInlines.h"
#include "FillLayer.h"
#include "RenderStyleInlines.h"
#include "SVGURIReference.h"
#include "Settings.h"
#include "ShapeValue.h"
#include "StyleImage.h"
#include "StyleReflection.h"
#include "TransformOperationsBuilder.h"

namespace WebCore {
namespace Style {

// <https://html.spec.whatwg.org/multipage/urls-and-fetching.html#cors-settings-attributes>
enum class LoadPolicy { CORS, NoCORS, Anonymous };
static void loadPendingImage(Document& document, const StyleImage* styleImage, const Element* element, LoadPolicy loadPolicy = LoadPolicy::NoCORS)
{
    if (!styleImage || !styleImage->isPending())
        return;

    bool isInUserAgentShadowTree = element && element->isInUserAgentShadowTree();
    ResourceLoaderOptions options = CachedResourceLoader::defaultCachedResourceOptions();
    options.contentSecurityPolicyImposition = isInUserAgentShadowTree ? ContentSecurityPolicyImposition::SkipPolicyCheck : ContentSecurityPolicyImposition::DoPolicyCheck;

    if (!isInUserAgentShadowTree && document.settings().useAnonymousModeWhenFetchingMaskImages()) {
        switch (loadPolicy) {
        case LoadPolicy::Anonymous:
            options.storedCredentialsPolicy = StoredCredentialsPolicy::DoNotUse;
            FALLTHROUGH;
        case LoadPolicy::CORS:
            options.mode = FetchOptions::Mode::Cors;
            options.credentials = FetchOptions::Credentials::SameOrigin;
            options.sameOriginDataURLFlag = SameOriginDataURLFlag::Set;
            break;
        case LoadPolicy::NoCORS:
            break;
        }
    }

    const_cast<StyleImage&>(*styleImage).load(document.cachedResourceLoader(), options);
}

void loadPendingResources(RenderStyle& style, Document& document, const Element* element)
{
    for (auto* backgroundLayer = &style.backgroundLayers(); backgroundLayer; backgroundLayer = backgroundLayer->next())
        loadPendingImage(document, backgroundLayer->image(), element);

    for (auto* contentData = style.contentData(); contentData; contentData = contentData->next()) {
        if (auto* imageContentData = dynamicDowncast<ImageContentData>(*contentData)) {
            auto& styleImage = imageContentData->image();
            loadPendingImage(document, &styleImage, element);
        }
    }

    if (auto* cursorList = style.cursors()) {
        for (size_t i = 0; i < cursorList->size(); ++i)
            loadPendingImage(document, cursorList->at(i).image(), element);
    }

    loadPendingImage(document, style.listStyleImage(), element);
    loadPendingImage(document, style.borderImageSource(), element);
    loadPendingImage(document, style.maskBorderSource(), element);

    if (auto* reflection = style.boxReflect())
        loadPendingImage(document, reflection->mask().image(), element);

    // Masking operations may be sensitive to timing attacks that can be used to reveal the pixel data of
    // the image used as the mask. As a means to mitigate such attacks CSS mask images and shape-outside
    // images are retrieved in "Anonymous" mode, which uses a potentially CORS-enabled fetch.
    for (auto* maskLayer = &style.maskLayers(); maskLayer; maskLayer = maskLayer->next())
        loadPendingImage(document, maskLayer->image(), element, LoadPolicy::CORS);

    if (style.shapeOutside())
        loadPendingImage(document, style.shapeOutside()->image(), element, LoadPolicy::Anonymous);

    // Are there other pseudo-elements that need resource loading? 
    if (auto* firstLineStyle = style.getCachedPseudoStyle({ PseudoId::FirstLine }))
        loadPendingResources(*firstLineStyle, document, element);
}

}
}
