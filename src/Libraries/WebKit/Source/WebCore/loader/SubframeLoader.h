/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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

#include "FrameLoader.h"
#include <wtf/WeakRef.h>

namespace WebCore {

class Document;
class HTMLFrameOwnerElement;
class HTMLMediaElement;
class HTMLPlugInImageElement;
class IntSize;
class LocalFrame;
class Widget;

// This is a slight misnomer. It handles the higher level logic of loading both subframes and plugins.
class FrameLoader::SubframeLoader {
    WTF_MAKE_NONCOPYABLE(SubframeLoader); WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    explicit SubframeLoader(LocalFrame&);

    void clear();

    void createFrameIfNecessary(HTMLFrameOwnerElement&, const AtomString& frameName);
    bool requestFrame(HTMLFrameOwnerElement&, const String& url, const AtomString& frameName, LockHistory = LockHistory::Yes, LockBackForwardList = LockBackForwardList::Yes);
    bool requestObject(HTMLPlugInImageElement&, const String& url, const AtomString& frameName,
        const String& serviceType, const Vector<AtomString>& paramNames, const Vector<AtomString>& paramValues);

    bool containsPlugins() const { return m_containsPlugins; }
    
    bool resourceWillUsePlugin(const String& url, const String& mimeType);

private:
    bool requestPlugin(HTMLPlugInImageElement&, const URL&, const String& serviceType, const Vector<AtomString>& paramNames, const Vector<AtomString>& paramValues, bool useFallback);
    LocalFrame* loadOrRedirectSubframe(HTMLFrameOwnerElement&, const URL&, const AtomString& frameName, LockHistory, LockBackForwardList);
    RefPtr<LocalFrame> loadSubframe(HTMLFrameOwnerElement&, const URL&, const AtomString& name, const URL& referrer);
    bool loadPlugin(HTMLPlugInImageElement&, const URL&, const String& mimeType, const Vector<AtomString>& paramNames, const Vector<AtomString>& paramValues, bool useFallback);

    bool shouldUsePlugin(const URL&, const String& mimeType, bool hasFallback, bool& useFallback);
    bool pluginIsLoadable(const URL&, const HTMLPlugInImageElement&, const String& mimeType) const;

    URL completeURL(const String&) const;

    bool shouldConvertInvalidURLsToBlank() const;
    Ref<LocalFrame> protectedFrame() const;

    bool canCreateSubFrame() const;

    bool m_containsPlugins { false };
    WeakRef<LocalFrame> m_frame;
};

} // namespace WebCore
