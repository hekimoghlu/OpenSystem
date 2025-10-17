/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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
#include "HTMLHtmlElement.h"

#include "Document.h"
#include "DocumentLoader.h"
#include "DocumentParser.h"
#include "ElementInlines.h"
#include "FrameDestructionObserverInlines.h"
#include "FrameLoader.h"
#include "HTMLNames.h"
#include "LocalFrame.h"
#include "Logging.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLHtmlElement);

using namespace HTMLNames;

HTMLHtmlElement::HTMLHtmlElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(htmlTag));
}

Ref<HTMLHtmlElement> HTMLHtmlElement::create(Document& document)
{
    return adoptRef(*new HTMLHtmlElement(htmlTag, document));
}

Ref<HTMLHtmlElement> HTMLHtmlElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLHtmlElement(tagName, document));
}

} // namespace WebCore
