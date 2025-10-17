/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#include "InputMethodState.h"
#include "MessageSenderInlines.h"
#include "PageClientImpl.h"
#include "WebKitUserMessage.h"
#include "WebKitWebViewBasePrivate.h"
#include "WebKitWebViewPrivate.h"
#include "WebPageMessages.h"
#include "WebPasteboardProxy.h"
#include "WebProcessProxy.h"
#include <WebCore/PlatformDisplay.h>
#include <WebCore/PlatformEvent.h>
#include <wtf/CallbackAggregator.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/glib/Sandbox.h>

namespace WebKit {

void WebPageProxy::platformInitialize()
{
}

GtkWidget* WebPageProxy::viewWidget()
{
    RefPtr pageClient = this->pageClient();
    return pageClient ? static_cast<PageClientImpl&>(*pageClient).viewWidget() : nullptr;
}

void WebPageProxy::bindAccessibilityTree(const String& plugID)
{
#if USE(GTK4)
    if (!isInsideFlatpak() || checkFlatpakPortalVersion(7))
        webkitWebViewBaseSetPlugID(WEBKIT_WEB_VIEW_BASE(viewWidget()), plugID);
#else
    auto* accessible = gtk_widget_get_accessible(viewWidget());
    atk_socket_embed(ATK_SOCKET(accessible), const_cast<char*>(plugID.utf8().data()));
    atk_object_notify_state_change(accessible, ATK_STATE_TRANSIENT, FALSE);
#endif
}

void WebPageProxy::didUpdateEditorState(const EditorState&, const EditorState& newEditorState)
{
    if (newEditorState.shouldIgnoreSelectionChanges)
        return;
    if (newEditorState.selectionIsRange)
        WebPasteboardProxy::singleton().setPrimarySelectionOwner(focusedFrame());
    if (RefPtr pageClient = this->pageClient())
        pageClient->selectionDidChange();
}

void WebPageProxy::setInputMethodState(std::optional<InputMethodState>&& state)
{
    webkitWebViewBaseSetInputMethodState(WEBKIT_WEB_VIEW_BASE(viewWidget()), WTFMove(state));
}

void WebPageProxy::showEmojiPicker(const WebCore::IntRect& caretRect, CompletionHandler<void(String)>&& completionHandler)
{
    webkitWebViewBaseShowEmojiChooser(WEBKIT_WEB_VIEW_BASE(viewWidget()), caretRect, WTFMove(completionHandler));
}

void WebPageProxy::showValidationMessage(const WebCore::IntRect& anchorClientRect, const String& message)
{
    RefPtr pageClient = this->pageClient();
    if (!pageClient)
        return;

    m_validationBubble = pageClient->createValidationBubble(message, { m_preferences->minimumFontSize() });
    m_validationBubble->showRelativeTo(anchorClientRect);
}

void WebPageProxy::sendMessageToWebViewWithReply(UserMessage&& message, CompletionHandler<void(UserMessage&&)>&& completionHandler)
{
    if (!WEBKIT_IS_WEB_VIEW(viewWidget())) {
        completionHandler(UserMessage(message.name, WEBKIT_USER_MESSAGE_UNHANDLED_MESSAGE));
        return;
    }

    webkitWebViewDidReceiveUserMessage(WEBKIT_WEB_VIEW(viewWidget()), WTFMove(message), WTFMove(completionHandler));
}

void WebPageProxy::sendMessageToWebView(UserMessage&& message)
{
    sendMessageToWebViewWithReply(WTFMove(message), [](UserMessage&&) { });
}

void WebPageProxy::accentColorDidChange()
{
    if (!hasRunningProcess() || !pageClient())
        return;

    auto accentColor = pageClient()->accentColor();
    legacyMainFrameProcess().send(Messages::WebPage::SetAccentColor(accentColor), webPageIDInMainFrameProcess());
}

OptionSet<WebCore::PlatformEvent::Modifier> WebPageProxy::currentStateOfModifierKeys()
{
#if USE(GTK4)
    auto* device = gdk_seat_get_keyboard(gdk_display_get_default_seat(gtk_widget_get_display(viewWidget())));
    auto gdkModifiers = gdk_device_get_modifier_state(device);
    bool capsLockActive = gdk_device_get_caps_lock_state(device);
#else
    auto* keymap = gdk_keymap_get_for_display(gtk_widget_get_display(viewWidget()));
    auto gdkModifiers = gdk_keymap_get_modifier_state(keymap);
    bool capsLockActive = gdk_keymap_get_caps_lock_state(keymap);
#endif

    OptionSet<WebCore::PlatformEvent::Modifier> modifiers;
    if (gdkModifiers & GDK_SHIFT_MASK)
        modifiers.add(WebCore::PlatformEvent::Modifier::ShiftKey);
    if (gdkModifiers & GDK_CONTROL_MASK)
        modifiers.add(WebCore::PlatformEvent::Modifier::ControlKey);
    if (gdkModifiers & GDK_MOD1_MASK)
        modifiers.add(WebCore::PlatformEvent::Modifier::AltKey);
    if (gdkModifiers & GDK_META_MASK)
        modifiers.add(WebCore::PlatformEvent::Modifier::MetaKey);
    if (capsLockActive)
        modifiers.add(WebCore::PlatformEvent::Modifier::CapsLockKey);
    return modifiers;
}

void WebPageProxy::callAfterNextPresentationUpdate(CompletionHandler<void()>&& callback)
{
    if (!hasRunningProcess() || !m_drawingArea) {
        callback();
        return;
    }

    Ref aggregator = CallbackAggregator::create([weakThis = WeakPtr { *this }, callback = WTFMove(callback)]() mutable {
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis)
            return callback();
        webkitWebViewBaseCallAfterNextPresentationUpdate(WEBKIT_WEB_VIEW_BASE(protectedThis->viewWidget()), WTFMove(callback));
    });
    auto drawingAreaIdentifier = m_drawingArea->identifier();
    forEachWebContentProcess([&] (auto& process, auto) {
        process.sendWithAsyncReply(Messages::DrawingArea::DispatchAfterEnsuringDrawing(), [aggregator] { }, drawingAreaIdentifier);
    });
}

} // namespace WebKit
