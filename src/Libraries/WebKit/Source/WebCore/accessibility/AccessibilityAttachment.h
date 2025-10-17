/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
#pragma once

#if ENABLE(ATTACHMENT_ELEMENT)

#include "AccessibilityRenderObject.h"

namespace WebCore {
    
class HTMLAttachmentElement;
class RenderAttachment;
    
class AccessibilityAttachment final : public AccessibilityRenderObject {
public:
    static Ref<AccessibilityAttachment> create(AXID, RenderAttachment&);
    HTMLAttachmentElement* attachmentElement() const;
    bool hasProgress(float* progress = nullptr) const;
    
private:
    explicit AccessibilityAttachment(AXID, RenderAttachment&);

    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::Button; }

    bool isAttachmentElement() const final { return true; }

    String roleDescription() const final;
    float valueForRange() const final;
    bool computeIsIgnored() const final;
    void accessibilityText(Vector<AccessibilityText>&) const final;
};
    
} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilityAttachment) \
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isAttachmentElement(); } \
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(ATTACHMENT_ELEMENT)
