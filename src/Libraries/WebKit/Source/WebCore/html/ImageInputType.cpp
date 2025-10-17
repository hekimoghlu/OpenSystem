/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
#include "ImageInputType.h"

#include "CachedImage.h"
#include "DOMFormData.h"
#include "ElementInlines.h"
#include "HTMLFormElement.h"
#include "HTMLImageLoader.h"
#include "HTMLInputElement.h"
#include "HTMLNames.h"
#include "HTMLParserIdioms.h"
#include "InputTypeNames.h"
#include "MouseEvent.h"
#include "RenderBoxInlines.h"
#include "RenderElementInlines.h"
#include "RenderImage.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ImageInputType);

using namespace HTMLNames;

ImageInputType::ImageInputType(HTMLInputElement& element)
    : BaseButtonInputType(Type::Image, element)
{
}

const AtomString& ImageInputType::formControlType() const
{
    return InputTypeNames::image();
}

bool ImageInputType::isFormDataAppendable() const
{
    return true;
}

bool ImageInputType::appendFormData(DOMFormData& formData) const
{
    ASSERT(element());
    if (!element()->isActivatedSubmit())
        return false;

    auto& name = protectedElement()->name();
    if (name.isEmpty()) {
        formData.append("x"_s, String::number(m_clickLocation.x()));
        formData.append("y"_s, String::number(m_clickLocation.y()));
        return true;
    }

    formData.append(makeString(name, ".x"_s), String::number(m_clickLocation.x()));
    formData.append(makeString(name, ".y"_s), String::number(m_clickLocation.y()));

    return true;
}

void ImageInputType::handleDOMActivateEvent(Event& event)
{
    ASSERT(element());
    Ref<HTMLInputElement> protectedElement(*element());
    if (protectedElement->isDisabledFormControl() || !protectedElement->form())
        return;

    Ref<HTMLFormElement> protectedForm(*protectedElement->form());

    m_clickLocation = IntPoint();
    if (event.underlyingEvent()) {
        Event& underlyingEvent = *event.underlyingEvent();
        if (auto* mouseEvent = dynamicDowncast<MouseEvent>(underlyingEvent)) {
            if (!mouseEvent->isSimulated())
                m_clickLocation = IntPoint(mouseEvent->offsetX(), mouseEvent->offsetY());
        }
    }

    // Update layout before processing form actions in case the style changes
    // the Form or button relationships.
    protectedElement->protectedDocument()->updateLayoutIgnorePendingStylesheets();

    if (RefPtr currentForm = protectedElement->form())
        currentForm->submitIfPossible(&event, element()); // Event handlers can run.

    event.setDefaultHandled();
}

RenderPtr<RenderElement> ImageInputType::createInputRenderer(RenderStyle&& style)
{
    ASSERT(element());
    return createRenderer<RenderImage>(RenderObject::Type::Image, *element(), WTFMove(style));
}

void ImageInputType::attributeChanged(const QualifiedName& name)
{
    if (name == altAttr) {
        if (RefPtr element = this->element()) {
            auto* renderer = element->renderer();
            if (auto* renderImage = dynamicDowncast<RenderImage>(renderer))
                renderImage->updateAltText();
        }
    } else if (name == srcAttr) {
        if (RefPtr element = this->element()) {
            if (element->renderer())
                element->ensureImageLoader().updateFromElementIgnoringPreviousError();
        }
    }
    BaseButtonInputType::attributeChanged(name);
}

void ImageInputType::attach()
{
    BaseButtonInputType::attach();

    ASSERT(element());
    HTMLImageLoader& imageLoader = protectedElement()->ensureImageLoader();
    imageLoader.updateFromElement();

    auto* renderer = downcast<RenderImage>(element()->renderer());
    if (!renderer)
        return;

    if (imageLoader.hasPendingBeforeLoadEvent())
        return;

    auto& imageResource = renderer->imageResource();
    imageResource.setCachedImage(imageLoader.protectedImage());

    // If we have no image at all because we have no src attribute, set
    // image height and width for the alt text instead.
    if (!imageLoader.image() && !imageResource.cachedImage())
        renderer->setImageSizeForAltText();
}

bool ImageInputType::shouldRespectAlignAttribute()
{
    return true;
}

bool ImageInputType::canBeSuccessfulSubmitButton()
{
    return true;
}

bool ImageInputType::shouldRespectHeightAndWidthAttributes()
{
    return true;
}

unsigned ImageInputType::height() const
{
    ASSERT(element());
    Ref<HTMLInputElement> element(*this->element());

    element->protectedDocument()->updateLayout({ LayoutOptions::ContentVisibilityForceLayout }, element.ptr());

    if (auto* renderer = element->renderer())
        return adjustForAbsoluteZoom(downcast<RenderBox>(*renderer).contentBoxHeight(), *renderer);

    // Check the attribute first for an explicit pixel value.
    if (auto optionalHeight = parseHTMLNonNegativeInteger(element->attributeWithoutSynchronization(heightAttr)))
        return optionalHeight.value();

    // If the image is available, use its height.
    auto* imageLoader = element->imageLoader();
    if (imageLoader && imageLoader->image())
        return imageLoader->image()->imageSizeForRenderer(element->renderer(), 1).height().toUnsigned();

    return 0;
}

unsigned ImageInputType::width() const
{
    ASSERT(element());
    Ref<HTMLInputElement> element(*this->element());

    element->protectedDocument()->updateLayout({ LayoutOptions::ContentVisibilityForceLayout }, element.ptr());

    if (auto* renderer = element->renderer())
        return adjustForAbsoluteZoom(downcast<RenderBox>(*renderer).contentBoxWidth(), *renderer);

    // Check the attribute first for an explicit pixel value.
    if (auto optionalWidth = parseHTMLNonNegativeInteger(element->attributeWithoutSynchronization(widthAttr)))
        return optionalWidth.value();

    // If the image is available, use its width.
    auto* imageLoader = element->imageLoader();
    if (imageLoader && imageLoader->image())
        return imageLoader->image()->imageSizeForRenderer(element->renderer(), 1).width().toUnsigned();

    return 0;
}

String ImageInputType::resultForDialogSubmit() const
{
    return makeString(m_clickLocation.x(), ',', m_clickLocation.y());
}

bool ImageInputType::dirAutoUsesValue() const
{
    return false;
}

} // namespace WebCore
