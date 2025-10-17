/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#include "RemoveFormatCommand.h"

#include "ApplyStyleCommand.h"
#include "Element.h"
#include "FrameSelection.h"
#include "HTMLNames.h"
#include "LocalFrame.h"
#include "MutableStyleProperties.h"
#include "NodeName.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/RobinHoodHashSet.h>

namespace WebCore {

using namespace HTMLNames;

RemoveFormatCommand::RemoveFormatCommand(Ref<Document>&& document)
    : CompositeEditCommand(WTFMove(document))
{
}

static bool isElementForRemoveFormatCommand(const Element* element)
{
    using namespace ElementNames;

    switch (element->elementName()) {
    case HTML::acronym:
    case HTML::b:
    case HTML::bdo:
    case HTML::big:
    case HTML::cite:
    case HTML::code:
    case HTML::dfn:
    case HTML::em:
    case HTML::font:
    case HTML::i:
    case HTML::ins:
    case HTML::kbd:
    case HTML::nobr:
    case HTML::q:
    case HTML::s:
    case HTML::samp:
    case HTML::small_:
    case HTML::strike:
    case HTML::strong:
    case HTML::sub:
    case HTML::sup:
    case HTML::tt:
    case HTML::u:
    case HTML::var:
        return true;
    default:
        break;
    }
    return false;
}

void RemoveFormatCommand::doApply()
{
    if (endingSelection().isNoneOrOrphaned())
        return;

    // Get the default style for this editable root, it's the style that we'll give the
    // content that we're operating on.
    auto defaultStyle = EditingStyle::create(endingSelection().rootEditableElement());

    // We want to remove everything but transparent background.
    // FIXME: We shouldn't access style().
    defaultStyle->style()->setProperty(CSSPropertyBackgroundColor, CSSValueTransparent);

    applyCommandToComposite(ApplyStyleCommand::create(protectedDocument(), defaultStyle.ptr(), isElementForRemoveFormatCommand, editingAction()));
}

}
