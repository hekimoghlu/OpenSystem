/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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
#include "AccessibilityAttachment.h"

#include "HTMLAttachmentElement.h"
#include "HTMLNames.h"
#include "LocalizedStrings.h"
#include "RenderAttachment.h"

#if ENABLE(ATTACHMENT_ELEMENT)

namespace WebCore {
    
using namespace HTMLNames;

AccessibilityAttachment::AccessibilityAttachment(AXID axID, RenderAttachment& renderer)
    : AccessibilityRenderObject(axID, renderer)
{
}

Ref<AccessibilityAttachment> AccessibilityAttachment::create(AXID axID, RenderAttachment& renderer)
{
    return adoptRef(*new AccessibilityAttachment(axID, renderer));
}

bool AccessibilityAttachment::hasProgress(float* progress) const
{
    auto& progressString = getAttribute(progressAttr);
    bool validProgress;
    float result = std::max<float>(std::min<float>(progressString.toFloat(&validProgress), 1), 0);
    if (progress)
        *progress = result;
    return validProgress;
}

float AccessibilityAttachment::valueForRange() const
{
    float progress = 0;
    hasProgress(&progress);
    return progress;
}
    
HTMLAttachmentElement* AccessibilityAttachment::attachmentElement() const
{
    ASSERT(is<HTMLAttachmentElement>(node()));
    return dynamicDowncast<HTMLAttachmentElement>(node());
}
    
String AccessibilityAttachment::roleDescription() const
{
    return AXAttachmentRoleText();
}

bool AccessibilityAttachment::computeIsIgnored() const
{
    return false;
}
    
void AccessibilityAttachment::accessibilityText(Vector<AccessibilityText>& textOrder) const
{
    RefPtr attachmentElement = this->attachmentElement();
    if (!attachmentElement)
        return;
    
    auto title = attachmentElement->attachmentTitle();
    auto& subtitle = attachmentElement->attachmentSubtitle();
    auto& action = getAttribute(actionAttr);
    
    if (action.length())
        textOrder.append(AccessibilityText(action, AccessibilityTextSource::Action));

    if (title.length())
        textOrder.append(AccessibilityText(title, AccessibilityTextSource::Title));

    if (subtitle.length())
        textOrder.append(AccessibilityText(subtitle, AccessibilityTextSource::Subtitle));
}

} // namespace WebCore

#endif // ENABLE(ATTACHMENT_ELEMENT)

