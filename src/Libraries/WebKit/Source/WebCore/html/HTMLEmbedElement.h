/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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

#include "HTMLPlugInImageElement.h"

namespace WebCore {

class HTMLEmbedElement final : public HTMLPlugInImageElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLEmbedElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLEmbedElement);
public:
    static Ref<HTMLEmbedElement> create(Document&);
    static Ref<HTMLEmbedElement> create(const QualifiedName&, Document&);

private:
    HTMLEmbedElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) final;

    bool rendererIsNeeded(const RenderStyle&) final;

    bool isURLAttribute(const Attribute&) const final;
    const AtomString& imageSourceURL() const final;

    bool isInteractiveContent() const final { return true; }

    RenderWidget* renderWidgetLoadingPlugin() const final;

    void updateWidget(CreatePlugins) final;

    void addSubresourceAttributeURLs(ListHashSet<URL>&) const final;

    void parametersForPlugin(Vector<AtomString>& paramNames, Vector<AtomString>& paramValues);
};

} // namespace WebCore
