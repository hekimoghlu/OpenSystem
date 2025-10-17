/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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
#include "ImageControlsMac.h"

#include "Chrome.h"
#include "ChromeClient.h"
#include "CommonAtomStrings.h"
#include "ContextMenuController.h"
#include "ElementInlines.h"
#include "ElementRareData.h"
#include "Event.h"
#include "EventHandler.h"
#include "EventLoop.h"
#include "EventNames.h"
#include "HTMLAttachmentElement.h"
#include "HTMLButtonElement.h"
#include "HTMLDivElement.h"
#include "HTMLImageElement.h"
#include "HTMLNames.h"
#include "HTMLStyleElement.h"
#include "MouseEvent.h"
#include "RenderAttachment.h"
#include "RenderImage.h"
#include "ShadowRoot.h"
#include "TreeScopeInlines.h"
#include "UserAgentParts.h"
#include "UserAgentStyleSheets.h"
#include <wtf/text/AtomString.h>

namespace WebCore {
namespace ImageControlsMac {

#if ENABLE(SERVICE_CONTROLS)

static const AtomString& imageControlsElementIdentifier()
{
    static MainThreadNeverDestroyed<const AtomString> identifier("image-controls"_s);
    return identifier;
}

static const AtomString& imageControlsButtonIdentifier()
{
    static MainThreadNeverDestroyed<const AtomString> identifier("image-controls-button"_s);
    return identifier;
}

bool hasImageControls(const HTMLElement& element)
{
    RefPtr shadowRoot = element.shadowRoot();
    if (!shadowRoot || shadowRoot->mode() != ShadowRootMode::UserAgent)
        return false;

    return shadowRoot->hasElementWithId(imageControlsElementIdentifier());
}

static RefPtr<HTMLElement> imageControlsHost(const Node& node)
{
    RefPtr host = dynamicDowncast<HTMLElement>(node.shadowHost());
    if (!host)
        return nullptr;

    return hasImageControls(*host) ? host : nullptr;
}

bool isImageControlsButtonElement(const Element& element)
{
    return imageControlsHost(element) && element.getIdAttribute() == imageControlsButtonIdentifier();
}

bool isInsideImageControls(const Node& node)
{
    RefPtr host = imageControlsHost(node);
    if (!host)
        return false;

    return host->protectedUserAgentShadowRoot()->contains(node);
}

void createImageControls(HTMLElement& element)
{
    Ref document = element.document();
    Ref shadowRoot = element.ensureUserAgentShadowRoot();
    Ref controlLayer = HTMLDivElement::create(document.get());
    controlLayer->setIdAttribute(imageControlsElementIdentifier());
    controlLayer->setAttributeWithoutSynchronization(HTMLNames::contenteditableAttr, falseAtom());
    shadowRoot->appendChild(controlLayer);
    
    static MainThreadNeverDestroyed<const String> shadowStyle(StringImpl::createWithoutCopying(imageControlsMacUserAgentStyleSheet));
    Ref style = HTMLStyleElement::create(HTMLNames::styleTag, document.get(), false);
    style->setTextContent(String { shadowStyle });
    shadowRoot->appendChild(WTFMove(style));
    
    Ref button = HTMLButtonElement::create(HTMLNames::buttonTag, element.document(), nullptr);
    button->setIdAttribute(imageControlsButtonIdentifier());
    controlLayer->appendChild(button);
    controlLayer->setUserAgentPart(UserAgentParts::appleAttachmentControlsContainer());
    
    if (CheckedPtr renderImage = dynamicDowncast<RenderImage>(element.renderer()))
        renderImage->setHasShadowControls(true);
}

static Image* imageFromImageElementNode(Node& node)
{
    CheckedPtr renderer = dynamicDowncast<RenderImage>(node.renderer());
    if (!renderer)
        return nullptr;
    CachedResourceHandle image = renderer->cachedImage();
    if (!image || image->errorOccurred())
        return nullptr;
    return image->imageForRenderer(renderer.get());
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

    if (ImageControlsMac::isImageControlsButtonElement(*target)) {
        CheckedPtr renderer = target->renderer();
        if (!renderer)
            return false;

        RefPtr view = frame->view();
        if (!view)
            return false;

        auto point = view->contentsToWindow(renderer->absoluteBoundingBoxRect()).minXMaxYCorner();

        if (RefPtr shadowHost = dynamicDowncast<HTMLImageElement>(target->shadowHost())) {
            RefPtr image = imageFromImageElementNode(*shadowHost);
            if (!image)
                return false;
            page->chrome().client().handleImageServiceClick(point, *image, *shadowHost);
        } else if (RefPtr shadowHost = dynamicDowncast<HTMLAttachmentElement>(target->shadowHost()))
            page->chrome().client().handlePDFServiceClick(point, *shadowHost);

        event.setDefaultHandled();
        return true;
    }
    return false;
}

static bool isImageMenuEnabled(HTMLElement& element)
{
    if (RefPtr imageElement = dynamicDowncast<HTMLImageElement>(element))
        return imageElement->isImageMenuEnabled();

    if (RefPtr attachmentElement = dynamicDowncast<HTMLAttachmentElement>(element))
        return attachmentElement->isImageMenuEnabled();

    return false;
}

void updateImageControls(HTMLElement& element)
{
    // If this is an image element and it is inside a shadow tree then it is part of an image control.
    if (element.isInShadowTree())
        return;

    if (!element.document().settings().imageControlsEnabled())
        return;

    element.protectedDocument()->checkedEventLoop()->queueTask(TaskSource::InternalAsyncTask, [weakElement = WeakPtr { element }] {
        RefPtr protectedElement = weakElement.get();
        if (!protectedElement)
            return;
        ASSERT(is<HTMLAttachmentElement>(*protectedElement) || is<HTMLImageElement>(*protectedElement));
        bool imageMenuEnabled = isImageMenuEnabled(*protectedElement);
        bool hasControls = hasImageControls(*protectedElement);
        if (!imageMenuEnabled && hasControls)
            destroyImageControls(*protectedElement);
        else if (imageMenuEnabled && !hasControls)
            tryCreateImageControls(*protectedElement);
    });
}

void tryCreateImageControls(HTMLElement& element)
{
    ASSERT(isImageMenuEnabled(element));
    ASSERT(!hasImageControls(element));
    createImageControls(element);
}

void destroyImageControls(HTMLElement& element)
{
    RefPtr shadowRoot = element.userAgentShadowRoot();
    if (!shadowRoot)
        return;

    if (RefPtr node = shadowRoot->firstChild()) {
        RefPtr htmlElement = dynamicDowncast<HTMLElement>(node.releaseNonNull());
        if (!htmlElement)
            return;
        RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(ImageControlsMac::hasImageControls(*htmlElement));
        shadowRoot->removeChild(*htmlElement);
    }

    auto* renderObject = element.renderer();
    if (!renderObject)
        return;

    if (CheckedPtr renderImage = dynamicDowncast<RenderImage>(*renderObject))
        renderImage->setHasShadowControls(false);
    else if (CheckedPtr renderAttachment = dynamicDowncast<RenderAttachment>(*renderObject))
        renderAttachment->setHasShadowControls(false);
}

#endif // ENABLE(SERVICE_CONTROLS)

} // namespace ImageControlsMac
} // namespace WebCore
