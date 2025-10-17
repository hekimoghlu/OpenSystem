/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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
#include "WebEditorClient.h"

#include <WebCore/Document.h>
#include <WebCore/Editor.h>
#include <WebCore/EventNames.h>
#include <WebCore/KeyboardEvent.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/PagePasteboardContext.h>
#include <WebCore/Pasteboard.h>
#include <WebCore/PlatformKeyboardEvent.h>
#include <WebCore/TextIterator.h>
#include <WebCore/markup.h>
#include <WebPage.h>
#include <variant>
#include <wtf/glib/GRefPtr.h>

namespace WebKit {
using namespace WebCore;

bool WebEditorClient::handleGtkEditorCommand(LocalFrame& frame, const String& command, bool allowTextInsertion)
{
    if (command == "GtkInsertEmoji"_s) {
        if (!allowTextInsertion)
            return false;
        m_page->showEmojiPicker(frame);
        return true;
    }

    return false;
}

bool WebEditorClient::executePendingEditorCommands(LocalFrame& frame, const Vector<WTF::String>& pendingEditorCommands, bool allowTextInsertion)
{
    Vector<std::variant<Editor::Command, String>> commands;
    for (auto& commandString : pendingEditorCommands) {
        if (commandString.startsWith("Gtk"_s))
            commands.append(commandString);
        else {
            Editor::Command command = frame.editor().command(commandString);
            if (command.isTextInsertion() && !allowTextInsertion)
                return false;

            commands.append(WTFMove(command));
        }
    }

    for (auto& commandVariant : commands) {
        if (std::holds_alternative<String>(commandVariant)) {
            if (!handleGtkEditorCommand(frame, std::get<String>(commandVariant), allowTextInsertion))
                return false;
        } else {
            auto& command = std::get<Editor::Command>(commandVariant);
            if (!command.execute())
                return false;
        }
    }

    return true;
}

void WebEditorClient::handleKeyboardEvent(KeyboardEvent& event)
{
    auto* platformEvent = event.underlyingPlatformEvent();
    if (!platformEvent)
        return;

    // If this was an IME event don't do anything.
    if (platformEvent->handledByInputMethod())
        return;

    ASSERT(event.target());
    auto* frame = downcast<Node>(event.target())->document().frame();
    ASSERT(frame);

    const Vector<String> pendingEditorCommands = platformEvent->commands();
    if (!pendingEditorCommands.isEmpty()) {

        // During RawKeyDown events if an editor command will insert text, defer
        // the insertion until the keypress event. We want keydown to bubble up
        // through the DOM first.
        if (platformEvent->type() == PlatformEvent::Type::RawKeyDown) {
            if (executePendingEditorCommands(*frame, pendingEditorCommands, false))
                event.setDefaultHandled();

            return;
        }

        // Only allow text insertion commands if the current node is editable.
        if (executePendingEditorCommands(*frame, pendingEditorCommands, frame->editor().canEdit())) {
            event.setDefaultHandled();
            return;
        }
    }

    // Don't allow text insertion for nodes that cannot edit.
    if (!frame->editor().canEdit())
        return;

    // This is just a normal text insertion, so wait to execute the insertion
    // until a keypress event happens. This will ensure that the insertion will not
    // be reflected in the contents of the field until the keyup DOM event.
    if (event.type() != eventNames().keypressEvent)
        return;

    // Don't insert null or control characters as they can result in unexpected behaviour
    if (event.charCode() < ' ')
        return;

    // Don't insert anything if a modifier is pressed
    if (platformEvent->controlKey() || platformEvent->altKey())
        return;

    if (frame->editor().insertText(platformEvent->text(), &event))
        event.setDefaultHandled();
}

void WebEditorClient::updateGlobalSelection(LocalFrame* frame)
{
    if (!frame->selection().isRange())
        return;
    auto range = frame->selection().selection().toNormalizedRange();
    if (!range)
        return;

    PasteboardWebContent pasteboardContent;
    pasteboardContent.canSmartCopyOrDelete = false;
    pasteboardContent.text = plainText(*range);
    pasteboardContent.markup = serializePreservingVisualAppearance(frame->selection().selection(), ResolveURLs::YesExcludingURLsForPrivacy);
    Pasteboard::createForGlobalSelection(PagePasteboardContext::create(frame->pageID()))->write(pasteboardContent);
}

bool WebEditorClient::shouldShowUnicodeMenu()
{
    return true;
}

}
