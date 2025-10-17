/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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

#include "HTMLPlugInElement.h"

namespace WebCore {

class HTMLImageLoader;

enum class CreatePlugins : bool { No, Yes };

// Base class for HTMLEmbedElement and HTMLObjectElement.
// FIXME: This is the only class that derives from HTMLPlugInElement, so we could merge the two classes.
class HTMLPlugInImageElement : public HTMLPlugInElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLPlugInImageElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLPlugInImageElement);
public:
    virtual ~HTMLPlugInImageElement();

    RenderEmbeddedObject* renderEmbeddedObject() const;

    virtual void updateWidget(CreatePlugins) = 0;

    const String& serviceType() const { return m_serviceType; }
    const String& url() const { return m_url; }

    bool needsWidgetUpdate() const { return m_needsWidgetUpdate; }
    void setNeedsWidgetUpdate(bool needsWidgetUpdate) { m_needsWidgetUpdate = needsWidgetUpdate; }

    bool shouldBypassCSPForPDFPlugin(const String& contentType) const;
    
protected:
    HTMLPlugInImageElement(const QualifiedName& tagName, Document&);

    void didMoveToNewDocument(Document& oldDocument, Document& newDocument) override;

    bool requestObject(const String& url, const String& mimeType, const Vector<AtomString>& paramNames, const Vector<AtomString>& paramValues) final;

    bool isImageType();
    HTMLImageLoader* imageLoader() { return m_imageLoader.get(); }
    void updateImageLoaderWithNewURLSoon();

    bool canLoadURL(const String& relativeURL) const;
    bool wouldLoadAsPlugIn(const String& relativeURL, const String& serviceType);

    void scheduleUpdateForAfterStyleResolution();

    String m_serviceType;
    String m_url;

private:
    bool isPlugInImageElement() const final { return true; }

    bool canLoadPlugInContent(const String& relativeURL, const String& mimeType) const;
    bool canLoadURL(const URL&) const;

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;
    bool childShouldCreateRenderer(const Node&) const override;
    void willRecalcStyle(Style::Change) final;
    void didRecalcStyle(Style::Change) final;
    void didAttachRenderers() final;
    void willDetachRenderers() final;

    void prepareForDocumentSuspension() final;
    void resumeFromDocumentSuspension() final;

    void updateAfterStyleResolution();

    bool m_needsWidgetUpdate { false };
    bool m_needsDocumentActivationCallbacks { false };
    std::unique_ptr<HTMLImageLoader> m_imageLoader;
    bool m_needsImageReload { false };
    bool m_hasUpdateScheduledForAfterStyleResolution { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::HTMLPlugInImageElement)
    static bool isType(const WebCore::HTMLPlugInElement& element) { return element.isPlugInImageElement(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* pluginElement = dynamicDowncast<WebCore::HTMLPlugInElement>(node);
        return pluginElement && isType(*pluginElement);
    }
SPECIALIZE_TYPE_TRAITS_END()
