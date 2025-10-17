/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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
#include "HTMLPictureElement.h"

#include "ElementChildIteratorInlines.h"
#include "HTMLAnchorElement.h"
#include "HTMLImageElement.h"
#include "ImageLoader.h"
#include "Logging.h"
#include "Settings.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLPictureElement);

HTMLPictureElement::HTMLPictureElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
}

HTMLPictureElement::~HTMLPictureElement() = default;

Ref<HTMLPictureElement> HTMLPictureElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLPictureElement(tagName, document));
}

void HTMLPictureElement::sourcesChanged()
{
    for (auto& element : childrenOfType<HTMLImageElement>(*this))
        element.selectImageSource(RelevantMutation::Yes);
}

void HTMLPictureElement::sourceDimensionAttributesChanged(const HTMLSourceElement& sourceElement)
{
    for (auto& element : childrenOfType<HTMLImageElement>(*this)) {
        if (&sourceElement == element.sourceElement())
            element.invalidateAttributeMapping();
    }
}

#if USE(SYSTEM_PREVIEW)
bool HTMLPictureElement::isSystemPreviewImage()
{
    if (!document().settings().systemPreviewEnabled())
        return false;

    auto* parent = dynamicDowncast<HTMLAnchorElement>(parentElement());
    return parent && parent->isSystemPreviewLink();
}
#endif

}

