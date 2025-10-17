/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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
#include "WebPage.h"

#include "MessageSenderInlines.h"
#include "WebFrame.h"
#include "WebKeyboardEvent.h"
#include "WebPageProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/BackForwardController.h>
#include <WebCore/Editor.h>
#include <WebCore/EventHandler.h>
#include <WebCore/FocusController.h>
#include <WebCore/KeyboardEvent.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/LocalFrameView.h>
#include <WebCore/NotImplemented.h>
#include <WebCore/Page.h>
#include <WebCore/PlatformKeyboardEvent.h>
#include <WebCore/PlatformScreen.h>
#include <WebCore/PointerCharacteristics.h>
#include <WebCore/RenderTheme.h>
#include <WebCore/RenderThemeAdwaita.h>
#include <WebCore/Settings.h>
#include <WebCore/SharedBuffer.h>
#include <WebCore/WindowsKeyboardCodes.h>
#include <gtk/gtk.h>
#include <wtf/glib/GUniquePtr.h>

namespace WebKit {
using namespace WebCore;

void WebPage::platformReinitialize()
{
}

bool WebPage::platformCanHandleRequest(const ResourceRequest&)
{
    notImplemented();
    return false;
}

bool WebPage::hoverSupportedByPrimaryPointingDevice() const
{
#if ENABLE(TOUCH_EVENTS)
    return !screenIsTouchPrimaryInputDevice();
#else
    return true;
#endif
}

bool WebPage::hoverSupportedByAnyAvailablePointingDevice() const
{
#if ENABLE(TOUCH_EVENTS)
    return !screenHasTouchDevice();
#else
    return true;
#endif
}

std::optional<PointerCharacteristics> WebPage::pointerCharacteristicsOfPrimaryPointingDevice() const
{
#if ENABLE(TOUCH_EVENTS)
    if (screenIsTouchPrimaryInputDevice())
        return PointerCharacteristics::Coarse;
#endif
    return PointerCharacteristics::Fine;
}

OptionSet<PointerCharacteristics> WebPage::pointerCharacteristicsOfAllAvailablePointingDevices() const
{
#if ENABLE(TOUCH_EVENTS)
    if (screenHasTouchDevice())
        return PointerCharacteristics::Coarse;
#endif
    return PointerCharacteristics::Fine;
}

void WebPage::collapseSelectionInFrame(FrameIdentifier frameID)
{
    WebFrame* frame = WebProcess::singleton().webFrame(frameID);
    if (!frame || !frame->coreLocalFrame())
        return;

    // Collapse the selection without clearing it.
    const VisibleSelection& selection = frame->coreLocalFrame()->selection().selection();
    frame->coreLocalFrame()->selection().setBase(selection.extent(), selection.affinity());
}

void WebPage::showEmojiPicker(LocalFrame& frame)
{
    CompletionHandler<void(String)> completionHandler = [frame = Ref { frame }](String result) {
        if (!result.isEmpty())
            frame->editor().insertText(result, nullptr);
    };
    sendWithAsyncReply(Messages::WebPageProxy::ShowEmojiPicker(frame.view()->contentsToRootView(frame.selection().absoluteCaretBounds())), WTFMove(completionHandler));
}

void WebPage::setAccentColor(WebCore::Color color)
{
    static_cast<RenderThemeAdwaita&>(RenderTheme::singleton()).setAccentColor(color);
}

} // namespace WebKit
