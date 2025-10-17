/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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
#include "WebPageProxy.h"

#include "DrawingAreaMessages.h"
#include "DrawingAreaProxy.h"
#include "EditorState.h"
#include "InputMethodState.h"
#include "PageClientImpl.h"
#include "UserMessage.h"
#include "WebProcessProxy.h"
#include <WebCore/PlatformEvent.h>
#include <wtf/CallbackAggregator.h>

#if USE(ATK)
#include <atk/atk.h>
#endif

#if ENABLE(WPE_PLATFORM)
#include <wpe/wpe-platform.h>
#endif

#if USE(GBM)
#include "DMABufRendererBufferFormat.h"
#endif

#if USE(GBM) && ENABLE(WPE_PLATFORM)
#include "MessageSenderInlines.h"
#include "WebPageMessages.h"
#endif

namespace WebKit {

void WebPageProxy::platformInitialize()
{
}

struct wpe_view_backend* WebPageProxy::viewBackend()
{
    RefPtr pageClient = this->pageClient();
    return pageClient ? static_cast<PageClientImpl&>(*pageClient).viewBackend() : nullptr;
}

#if ENABLE(WPE_PLATFORM)
WPEView* WebPageProxy::wpeView() const
{
    RefPtr pageClient = this->pageClient();
    return pageClient ? static_cast<PageClientImpl&>(*pageClient).wpeView() : nullptr;
}
#endif

void WebPageProxy::bindAccessibilityTree(const String& plugID)
{
#if USE(ATK)
    RefPtr pageClient = this->pageClient();
    if (!pageClient)
        return;
    auto* accessible = static_cast<PageClientImpl&>(*pageClient).accessible();
    atk_socket_embed(ATK_SOCKET(accessible), const_cast<char*>(plugID.utf8().data()));
    atk_object_notify_state_change(accessible, ATK_STATE_TRANSIENT, FALSE);
#else
    UNUSED_PARAM(plugID);
#endif
}

void WebPageProxy::didUpdateEditorState(const EditorState&, const EditorState& newEditorState)
{
    if (!newEditorState.shouldIgnoreSelectionChanges) {
        if (RefPtr pageClient = this->pageClient())
            pageClient->selectionDidChange();
    }
}

void WebPageProxy::sendMessageToWebViewWithReply(UserMessage&& message, CompletionHandler<void(UserMessage&&)>&& completionHandler)
{
    RefPtr pageClient = this->pageClient();
    if (!pageClient)
        return completionHandler({ });
    static_cast<PageClientImpl&>(*pageClient).sendMessageToWebView(WTFMove(message), WTFMove(completionHandler));
}

void WebPageProxy::sendMessageToWebView(UserMessage&& message)
{
    sendMessageToWebViewWithReply(WTFMove(message), [](UserMessage&&) { });
}

void WebPageProxy::setInputMethodState(std::optional<InputMethodState>&& state)
{
    if (RefPtr pageClient = this->pageClient())
        static_cast<PageClientImpl&>(*pageClient).setInputMethodState(WTFMove(state));
}

#if USE(GBM)
Vector<DMABufRendererBufferFormat> WebPageProxy::preferredBufferFormats() const
{
#if ENABLE(WPE_PLATFORM)
    auto* view = wpeView();
    if (!view)
        return { };

    auto* formats = wpe_view_get_preferred_dma_buf_formats(view);
    if (!formats)
        return { };

    Vector<DMABufRendererBufferFormat> dmabufFormats;
    const char* mainDevice = wpe_buffer_dma_buf_formats_get_device(formats);
    auto groupCount = wpe_buffer_dma_buf_formats_get_n_groups(formats);
    for (unsigned i = 0; i < groupCount; ++i) {
        DMABufRendererBufferFormat dmabufFormat;
        switch (wpe_buffer_dma_buf_formats_get_group_usage(formats, i)) {
        case WPE_BUFFER_DMA_BUF_FORMAT_USAGE_RENDERING:
            dmabufFormat.usage = DMABufRendererBufferFormat::Usage::Rendering;
            break;
        case WPE_BUFFER_DMA_BUF_FORMAT_USAGE_MAPPING:
            dmabufFormat.usage = DMABufRendererBufferFormat::Usage::Mapping;
            break;
        case WPE_BUFFER_DMA_BUF_FORMAT_USAGE_SCANOUT:
            dmabufFormat.usage = DMABufRendererBufferFormat::Usage::Scanout;
            break;
        }
        const char* targetDevice = wpe_buffer_dma_buf_formats_get_group_device(formats, i);
        dmabufFormat.drmDevice = targetDevice ? targetDevice : mainDevice;
        auto formatsCount = wpe_buffer_dma_buf_formats_get_group_n_formats(formats, i);
        dmabufFormat.formats.reserveInitialCapacity(formatsCount);
        for (unsigned j = 0; j < formatsCount; ++j) {
            DMABufRendererBufferFormat::Format format;
            format.fourcc = wpe_buffer_dma_buf_formats_get_format_fourcc(formats, i, j);
            auto* modifiers = wpe_buffer_dma_buf_formats_get_format_modifiers(formats, i, j);
            format.modifiers.reserveInitialCapacity(modifiers->len);
            for (unsigned k = 0; k < modifiers->len; ++k) {
                WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // WPE port
                auto* modifier = &g_array_index(modifiers, guint64, k);
                WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
                format.modifiers.append(*modifier);
            }
            dmabufFormat.formats.append(WTFMove(format));
        }
        dmabufFormats.append(WTFMove(dmabufFormat));
    }

    return dmabufFormats;
#else
    return { };
#endif
}

#if USE(GBM) && ENABLE(WPE_PLATFORM)
void WebPageProxy::preferredBufferFormatsDidChange()
{
    auto* view = wpeView();
    if (!view)
        return;

    legacyMainFrameProcess().send(Messages::WebPage::PreferredBufferFormatsDidChange(preferredBufferFormats()), webPageIDInMainFrameProcess());
}
#endif
#endif

OptionSet<WebCore::PlatformEvent::Modifier> WebPageProxy::currentStateOfModifierKeys()
{
#if ENABLE(WPE_PLATFORM)
    auto* view = wpeView();
    if (!view)
        return { };

    auto* keymap = wpe_display_get_keymap(wpe_view_get_display(view), nullptr);
    if (!keymap)
        return { };

    OptionSet<WebCore::PlatformEvent::Modifier> modifiers;
    auto wpeModifiers = wpe_keymap_get_modifiers(keymap);
    if (wpeModifiers & WPE_MODIFIER_KEYBOARD_CONTROL)
        modifiers.add(WebCore::PlatformEvent::Modifier::ControlKey);
    if (wpeModifiers & WPE_MODIFIER_KEYBOARD_SHIFT)
        modifiers.add(WebCore::PlatformEvent::Modifier::ShiftKey);
    if (wpeModifiers & WPE_MODIFIER_KEYBOARD_ALT)
        modifiers.add(WebCore::PlatformEvent::Modifier::AltKey);
    if (wpeModifiers & WPE_MODIFIER_KEYBOARD_META)
        modifiers.add(WebCore::PlatformEvent::Modifier::MetaKey);
    if (wpeModifiers & WPE_MODIFIER_KEYBOARD_CAPS_LOCK)
        modifiers.add(WebCore::PlatformEvent::Modifier::CapsLockKey);
    return modifiers;
#else
    return { };
#endif
}

void WebPageProxy::callAfterNextPresentationUpdate(CompletionHandler<void()>&& callback)
{
    if (!hasRunningProcess() || !m_drawingArea) {
        callback();
        return;
    }

#if USE(COORDINATED_GRAPHICS)
    Ref aggregator = CallbackAggregator::create([weakThis = WeakPtr { *this }, callback = WTFMove(callback)]() mutable {
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis)
            return callback();

        if (RefPtr pageClient = protectedThis->pageClient())
            static_cast<PageClientImpl&>(*pageClient).callAfterNextPresentationUpdate(WTFMove(callback));
    });
    auto drawingAreaIdentifier = m_drawingArea->identifier();
    forEachWebContentProcess([&] (auto& process, auto) {
        process.sendWithAsyncReply(Messages::DrawingArea::DispatchAfterEnsuringDrawing(), [aggregator] { }, drawingAreaIdentifier);
    });
#else
    callback();
#endif
}

} // namespace WebKit
