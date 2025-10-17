/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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

#include "PluginReplacement.h"
#include <wtf/HashMap.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class WeakPtrImplWithEventTargetData;
class YouTubeEmbedShadowElement;

class YouTubePluginReplacement final : public PluginReplacement {
public:
    static void registerPluginReplacement(PluginReplacementRegistrar);

    WEBCORE_EXPORT static AtomString youTubeURLFromAbsoluteURL(const URL& srcURL, const AtomString& srcString);

private:
    YouTubePluginReplacement(HTMLPlugInElement&, const Vector<AtomString>& paramNames, const Vector<AtomString>& paramValues);
    virtual ~YouTubePluginReplacement();

    static Ref<PluginReplacement> create(HTMLPlugInElement&, const Vector<AtomString>& paramNames, const Vector<AtomString>& paramValues);

    static bool supportsMIMEType(const String&);
    static bool supportsFileExtension(StringView);
    static bool supportsURL(const URL&);

    void installReplacement(ShadowRoot&) final;

    AtomString youTubeURL(const AtomString& rawURL);

    bool willCreateRenderer() final { return m_embedShadowElement; }
    RenderPtr<RenderElement> createElementRenderer(HTMLPlugInElement&, RenderStyle&&, const RenderTreePosition&) final;

    WeakPtr<HTMLPlugInElement, WeakPtrImplWithEventTargetData> m_parentElement;
    RefPtr<YouTubeEmbedShadowElement> m_embedShadowElement;
    HashMap<AtomString, AtomString> m_attributes;
};

}
