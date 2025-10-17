/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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

#if ENABLE(VIDEO)

#include "HTMLDocument.h"

namespace WebCore {

class MediaDocument final : public HTMLDocument {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MediaDocument);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MediaDocument);
public:
    static Ref<MediaDocument> create(LocalFrame* frame, const Settings& settings, const URL& url)
    {
        auto document = adoptRef(*new MediaDocument(frame, settings, url));
        document->addToContextsMap();
        return document;
    }
    virtual ~MediaDocument();

    void mediaElementNaturalSizeChanged(const IntSize&);
    String outgoingReferrer() const { return m_outgoingReferrer; }

private:
    MediaDocument(LocalFrame*, const Settings&, const URL&);

    Ref<DocumentParser> createParser() override;

    void defaultEventHandler(Event&) override { }

    void replaceMediaElementTimerFired();

    String m_outgoingReferrer;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MediaDocument)
    static bool isType(const WebCore::Document& document) { return document.isMediaDocument(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* document = dynamicDowncast<WebCore::Document>(node);
        return document && isType(*document);
    }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(VIDEO)
