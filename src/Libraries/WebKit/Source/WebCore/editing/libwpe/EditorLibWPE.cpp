/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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
#include "Editor.h"

#if USE(LIBWPE)

#include "DocumentFragment.h"
#include "FrameDestructionObserverInlines.h"
#include "LocalFrame.h"
#include "NotImplemented.h"
#include "Pasteboard.h"
#include "Settings.h"
#include "markup.h"

namespace WebCore {

static RefPtr<DocumentFragment> createFragmentFromPasteboardData(Pasteboard& pasteboard, LocalFrame& frame, const SimpleRange& range, bool allowPlainText, bool& chosePlainText)
{
    chosePlainText = false;

    Vector<String> types = pasteboard.typesForLegacyUnsafeBindings();
    if (types.isEmpty())
        return nullptr;

    if (types.contains("text/html;charset=utf-8"_s) && frame.document()) {
        String markup = pasteboard.readString("text/html;charset=utf-8"_s);
        return createFragmentFromMarkup(*frame.document(), markup, emptyString(), { });
    }

    if (!allowPlainText)
        return nullptr;

    if (types.contains("text/plain;charset=utf-8"_s)) {
        chosePlainText = true;
        return createFragmentFromText(range, pasteboard.readString("text/plain;charset=utf-8"_s));
    }

    return nullptr;
}

void Editor::writeSelectionToPasteboard(Pasteboard& pasteboard)
{
    PasteboardWebContent pasteboardContent;
    pasteboardContent.text = selectedTextForDataTransfer();
    pasteboardContent.markup = serializePreservingVisualAppearance(document().selection().selection(), ResolveURLs::YesExcludingURLsForPrivacy, SerializeComposedTree::Yes, IgnoreUserSelectNone::Yes);
    pasteboard.write(pasteboardContent);
}

void Editor::writeImageToPasteboard(Pasteboard&, Element&, const URL&, const String&)
{
    notImplemented();
}

void Editor::pasteWithPasteboard(Pasteboard* pasteboard, OptionSet<PasteOption> options)
{
    auto range = selectedRange();
    if (!range)
        return;

    bool chosePlainText;
    auto fragment = createFragmentFromPasteboardData(*pasteboard, *document().frame(), *range, options.contains(PasteOption::AllowPlainText), chosePlainText);

    if (fragment && options.contains(PasteOption::AsQuotation))
        quoteFragmentForPasting(*fragment);

    if (fragment && shouldInsertFragment(*fragment, *range, EditorInsertAction::Pasted))
        pasteAsFragment(*fragment, canSmartReplaceWithPasteboard(*pasteboard), chosePlainText, options.contains(PasteOption::IgnoreMailBlockquote) ? MailBlockquoteHandling::IgnoreBlockquote : MailBlockquoteHandling::RespectBlockquote);
}

void Editor::platformCopyFont()
{
}

void Editor::platformPasteFont()
{
}

} // namespace WebCore

#endif // USE(LIBWPE)
