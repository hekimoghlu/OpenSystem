/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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

#include "CachedFont.h"
#include "CachedFontClient.h"
#include "CachedResourceHandle.h"
#include "FontLoadRequest.h"
#include "FontSelectionAlgorithm.h"
#include "ScriptExecutionContext.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

class FontCreationContext;

class CachedFontLoadRequest final : public FontLoadRequest, public CachedFontClient {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    CachedFontLoadRequest(CachedFont& font, ScriptExecutionContext& context)
        : m_font(&font)
        , m_context(context)
    {
    }

    ~CachedFontLoadRequest()
    {
        if (m_fontLoadRequestClient)
            protectedCachedFont()->removeClient(*this);
    }

    CachedFont& cachedFont() const { return *m_font; }
    CachedResourceHandle<CachedFont> protectedCachedFont() const { return m_font; }

private:
    const URL& url() const final { return m_font->url(); }
    bool isPending() const final { return m_font->status() == CachedResource::Status::Pending; }
    bool isLoading() const final { return m_font->isLoading(); }
    bool errorOccurred() const final { return m_font->errorOccurred(); }

    bool ensureCustomFontData() final
    {
        bool result = m_font->ensureCustomFontData();
        if (!result && m_font->didRefuseToParseCustomFontWithSafeFontParser()) {
            if (RefPtr context = m_context.get()) {
                auto message = makeString("[Lockdown Mode] This font wasn't parsed: "_s, m_font->url().string());
                context->addConsoleMessage(MessageSource::Security, MessageLevel::Info, message);
            }
        }
        return result;
    }

    RefPtr<Font> createFont(const FontDescription& description, bool syntheticBold, bool syntheticItalic, const FontCreationContext& fontCreationContext) final
    {
        return protectedCachedFont()->createFont(description, syntheticBold, syntheticItalic, fontCreationContext);
    }

    void setClient(FontLoadRequestClient* client) final
    {
        WeakPtr oldClient = m_fontLoadRequestClient;
        m_fontLoadRequestClient = client;

        if (!client && oldClient)
            protectedCachedFont()->removeClient(*this);
        else if (client && !oldClient)
            protectedCachedFont()->addClient(*this);
    }

    bool isCachedFontLoadRequest() const final { return true; }

    void fontLoaded(CachedFont& font) final
    {
        if (m_fontLoadedProcessed)
            return;

        m_fontLoadedProcessed = true;
        ASSERT_UNUSED(font, &font == m_font.get());
        if (m_fontLoadRequestClient)
            m_fontLoadRequestClient->fontLoaded(*this); // fontLoaded() might destroy this object. Don't deref its members after it.
    }

    CachedResourceHandle<CachedFont> m_font;
    WeakPtr<FontLoadRequestClient> m_fontLoadRequestClient;
    WeakPtr<ScriptExecutionContext> m_context;
    bool m_fontLoadedProcessed { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FONTLOADREQUEST(WebCore::CachedFontLoadRequest, isCachedFontLoadRequest())
