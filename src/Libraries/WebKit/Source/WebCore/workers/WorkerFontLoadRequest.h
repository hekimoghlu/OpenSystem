/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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

#include "FontLoadRequest.h"
#include "ResourceLoaderOptions.h"
#include "SharedBuffer.h"
#include "ThreadableLoaderClient.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class FontCreationContext;
class ScriptExecutionContext;
class WorkerGlobalScope;

struct FontCustomPlatformData;

class WorkerFontLoadRequest final : public FontLoadRequest, public ThreadableLoaderClient {
    WTF_MAKE_TZONE_ALLOCATED(WorkerFontLoadRequest);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WorkerFontLoadRequest);
public:
    WorkerFontLoadRequest(URL&&, LoadedFromOpaqueSource);
    ~WorkerFontLoadRequest() = default;

    void load(WorkerGlobalScope&);

private:
    const URL& url() const final { return m_url; }
    bool isPending() const final { return !m_isLoading && !m_errorOccurred && !m_data; }
    bool isLoading() const final { return m_isLoading; }
    bool errorOccurred() const final { return m_errorOccurred; }

    bool ensureCustomFontData() final;
    RefPtr<Font> createFont(const FontDescription&, bool syntheticBold, bool syntheticItalic, const FontCreationContext&) final;

    void setClient(FontLoadRequestClient*) final;

    bool isWorkerFontLoadRequest() const final { return true; }

    void didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse&) final;
    void didReceiveData(const SharedBuffer&) final;
    void didFinishLoading(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const NetworkLoadMetrics&) final;
    void didFail(std::optional<ScriptExecutionContextIdentifier>, const ResourceError&) final;

    URL m_url;
    LoadedFromOpaqueSource m_loadedFromOpaqueSource;

    bool m_isLoading { false };
    bool m_notifyOnClientSet { false };
    bool m_errorOccurred { false };
    FontLoadRequestClient* m_fontLoadRequestClient { nullptr };

    WeakPtr<ScriptExecutionContext> m_context;
    SharedBufferBuilder m_data;
    RefPtr<FontCustomPlatformData> m_fontCustomPlatformData;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FONTLOADREQUEST(WebCore::WorkerFontLoadRequest, isWorkerFontLoadRequest())
