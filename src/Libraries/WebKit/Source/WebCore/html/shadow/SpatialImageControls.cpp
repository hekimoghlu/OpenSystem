/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
#include "SpatialImageControls.h"

#include "ElementInlines.h"
#include "ElementRareData.h"
#include "Event.h"
#include "EventLoop.h"
#include "HTMLButtonElement.h"
#include "HTMLDivElement.h"
#include "HTMLImageElement.h"
#include "HTMLNames.h"
#include "HTMLSpanElement.h"
#include "HTMLStyleElement.h"
#include "MouseEvent.h"
#include "RenderImage.h"
#include "ShadowRoot.h"
#include "TreeScopeInlines.h"
#include "UserAgentStyleSheets.h"
#include <wtf/text/AtomString.h>

namespace WebCore {
namespace SpatialImageControls {

#if ENABLE(SPATIAL_IMAGE_CONTROLS)

static const AtomString& spatialImageControlsElementIdentifier()
{
    static MainThreadNeverDestroyed<const AtomString> identifier("spatial-image-controls"_s);
    return identifier;
}

static const AtomString& spatialImageControlsButtonIdentifier()
{
    static MainThreadNeverDestroyed<const AtomString> identifier("spatial-image-controls-button"_s);
    return identifier;
}

bool hasSpatialImageControls(const HTMLElement& element)
{
    RefPtr shadowRoot = element.shadowRoot();
    if (!shadowRoot || shadowRoot->mode() != ShadowRootMode::UserAgent)
        return false;

    return shadowRoot->hasElementWithId(spatialImageControlsElementIdentifier());
}

static RefPtr<HTMLElement> spatialImageControlsHost(const Node& node)
{
    RefPtr host = dynamicDowncast<HTMLElement>(node.shadowHost());
    if (!host)
        return nullptr;

    return hasSpatialImageControls(*host) ? host : nullptr;
}

bool isSpatialImageControlsButtonElement(const Element& element)
{
    return spatialImageControlsHost(element) && element.getIdAttribute() == spatialImageControlsButtonIdentifier();
}

bool shouldHaveSpatialControls(HTMLImageElement& element)
{
    if (!element.document().settings().spatialImageControlsEnabled())
        return false;

    bool hasSpatialcontrolsAttribute = element.hasAttributeWithoutSynchronization(HTMLNames::controlsAttr);

    auto* cachedImage = element.cachedImage();
    if (!cachedImage)
        return false;

    auto* image = cachedImage->image();
    if (!image)
        return false;

    return hasSpatialcontrolsAttribute && image->isSpatial();
}

void ensureSpatialControls(HTMLImageElement& imageElement)
{
    if (!shouldHaveSpatialControls(imageElement) || hasSpatialImageControls(imageElement))
        return;

    imageElement.protectedDocument()->checkedEventLoop()->queueTask(TaskSource::InternalAsyncTask, [weakElement = WeakPtr { imageElement }] {
        RefPtr element = weakElement.get();
        if (!element)
            return;
        Ref shadowRoot = element->ensureUserAgentShadowRoot();
        Ref document = element->document();

        if (hasSpatialImageControls(*element))
            return;

        double paddingValue = 20;
        unsigned imageHeight = element->height();
        if (imageHeight >= 400 && imageHeight < 490)
            paddingValue = 24;
        else if (imageHeight >= 490)
            paddingValue = 28;

        Ref controlLayer = HTMLDivElement::create(document.get());
        controlLayer->setIdAttribute(spatialImageControlsElementIdentifier());
        controlLayer->setAttributeWithoutSynchronization(HTMLNames::contenteditableAttr, falseAtom());
        controlLayer->setInlineStyleProperty(CSSPropertyDisplay, "flex"_s);
        controlLayer->setInlineStyleProperty(CSSPropertyFlexDirection, "column"_s);
        controlLayer->setInlineStyleProperty(CSSPropertyJustifyContent, "space-between"_s);
        controlLayer->setInlineStyleProperty(CSSPropertyPosition, "relative"_s);
        controlLayer->setInlineStyleProperty(CSSPropertyBoxSizing, "border-box"_s);
        controlLayer->setInlineStyleProperty(CSSPropertyPadding, paddingValue, CSSUnitType::CSS_PX);
        shadowRoot->appendChild(controlLayer);

        static MainThreadNeverDestroyed<const String> shadowStyle(StringImpl::createWithoutCopying(spatialImageControlsUserAgentStyleSheet));
        Ref style = HTMLStyleElement::create(HTMLNames::styleTag, document.get(), false);
        style->setTextContent(String { shadowStyle });
        controlLayer->appendChild(WTFMove(style));

        Ref button = HTMLButtonElement::create(HTMLNames::buttonTag, document.get(), nullptr);
        button->setIdAttribute(spatialImageControlsButtonIdentifier());
        controlLayer->appendChild(button);

        Ref backgroundBlurLayer = HTMLDivElement::create(document.get());
        backgroundBlurLayer->setIdAttribute("background-tint"_s);
        controlLayer->appendChild(backgroundBlurLayer);

        Ref blur = HTMLDivElement::create(document.get());
        blur->setIdAttribute("blur"_s);
        backgroundBlurLayer->appendChild(blur);

        Ref tint = HTMLDivElement::create(document.get());
        tint->setIdAttribute("tint"_s);
        backgroundBlurLayer->appendChild(tint);

        Ref bottomGradient = HTMLDivElement::create(document.get());
        bottomGradient->setIdAttribute("bottom-gradient"_s);
        controlLayer->appendChild(bottomGradient);

        Ref bottomLabelText = HTMLDivElement::create(document.get());
        bottomLabelText->setIdAttribute("label"_s);
        bottomLabelText->setTextContent("SPATIAL"_s);
        controlLayer->appendChild(bottomLabelText);

        Ref glyphSpan = HTMLSpanElement::create(document.get());
        glyphSpan->setIdAttribute("spatial-glyph"_s);
        bottomLabelText->insertBefore(glyphSpan, bottomLabelText->protectedFirstChild());

        if (CheckedPtr renderImage = dynamicDowncast<RenderImage>(element->renderer()))
            renderImage->setHasShadowControls(true);
    });
}

bool handleEvent(HTMLElement& element, Event& event)
{
    if (!isAnyClick(event))
        return false;

    RefPtr frame = element.document().frame();
    if (!frame)
        return false;

    RefPtr page = element.document().page();
    if (!page)
        return false;

    RefPtr mouseEvent = dynamicDowncast<MouseEvent>(event);
    if (!mouseEvent)
        return false;

    RefPtr target = dynamicDowncast<Element>(mouseEvent->target());
    if (!target)
        return false;

    if (SpatialImageControls::isSpatialImageControlsButtonElement(*target)) {
        RefPtr img = dynamicDowncast<HTMLImageElement>(target->shadowHost());
        img->webkitRequestFullscreen();

        event.setDefaultHandled();
        return true;
    }
    return false;
}

void destroySpatialImageControls(HTMLElement& element)
{
    element.protectedDocument()->checkedEventLoop()->queueTask(TaskSource::InternalAsyncTask, [weakElement = WeakPtr { element }] {
        RefPtr protectedElement = weakElement.get();
        if (!protectedElement)
            return;
        RefPtr shadowRoot = protectedElement->userAgentShadowRoot();
        if (!shadowRoot)
            return;

        if (RefPtr element = shadowRoot->getElementById(spatialImageControlsElementIdentifier()))
            element->remove();

        auto* renderObject = protectedElement->renderer();
        if (!renderObject)
            return;

        if (CheckedPtr renderImage = dynamicDowncast<RenderImage>(*renderObject))
            renderImage->setHasShadowControls(false);
    });
}

void updateSpatialImageControls(HTMLImageElement& element)
{
    if (!element.document().settings().spatialImageControlsEnabled())
        return;

    bool shouldHaveControls = shouldHaveSpatialControls(element);
    bool hasControls = hasSpatialImageControls(element);

    if (shouldHaveControls && !hasControls)
        ensureSpatialControls(element);
    else if (!shouldHaveControls && hasControls)
        destroySpatialImageControls(element);
}

#endif

} // namespace SpatialImageControls
} // namespace WebCore
