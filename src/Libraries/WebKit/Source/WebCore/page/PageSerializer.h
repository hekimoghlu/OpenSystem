/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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

#include "SharedBuffer.h"
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/ListHashSet.h>
#include <wtf/URLHash.h>

namespace WebCore {

class CachedImage;
class CSSStyleSheet;
class Document;
class LocalFrame;
class Page;
class RenderElement;
class StyleProperties;
class StyleRule;

// This class is used to serialize a page contents back to text (typically HTML).
// It serializes all the page frames and retrieves resources such as images and CSS stylesheets.
class PageSerializer {
public:
    struct Resource {
        URL url;
        String mimeType;
        RefPtr<SharedBuffer> data;
    };

    explicit PageSerializer(Vector<Resource>&);

    // Initiates the serialization of the frame's page. All serialized content and retrieved
    // resources are added to the Vector passed to the constructor. The first resource in that
    // vector is the top frame serialized content.
    void serialize(Page&);

private:
    class SerializerMarkupAccumulator;

    URL urlForBlankFrame(LocalFrame*);

    void serializeFrame(LocalFrame*);

    // Serializes the stylesheet back to text and adds it to the resources if URL is not-empty.
    // It also adds any resources included in that stylesheet (including any imported stylesheets and their own resources).
    void serializeCSSStyleSheet(CSSStyleSheet*, const URL&);

    void addImageToResources(CachedImage*, RenderElement*, const URL&);
    void retrieveResourcesForProperties(const StyleProperties*, Document*);
    void retrieveResourcesForRule(StyleRule&, Document*);

    Vector<Resource>& m_resources;
    ListHashSet<URL> m_resourceURLs;
    UncheckedKeyHashMap<LocalFrame*, URL> m_blankFrameURLs;
    unsigned m_blankFrameCounter { 0 };
};

} // namespace WebCore
