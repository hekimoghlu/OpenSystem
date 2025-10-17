/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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

#include "ClipboardUtilitiesWin.h"
#include "DocumentFragment.h"
#include "FrameDestructionObserverInlines.h"
#include "FrameSelection.h"
#include "LocalFrame.h"
#include "Pasteboard.h"
#include "Range.h"
#include "windows.h"

namespace WebCore {

void Editor::pasteWithPasteboard(Pasteboard* pasteboard, OptionSet<PasteOption> options)
{
    auto range = selectedRange();
    if (!range)
        return;

    bool chosePlainText;
    auto fragment = pasteboard->documentFragment(*document().frame(), *range, options.contains(PasteOption::AllowPlainText), chosePlainText);

    if (fragment && options.contains(PasteOption::AsQuotation))
        quoteFragmentForPasting(*fragment);

    if (fragment && shouldInsertFragment(*fragment, *range, EditorInsertAction::Pasted))
        pasteAsFragment(fragment.releaseNonNull(), canSmartReplaceWithPasteboard(*pasteboard), chosePlainText, options.contains(PasteOption::IgnoreMailBlockquote) ? MailBlockquoteHandling::IgnoreBlockquote : MailBlockquoteHandling::RespectBlockquote);
}

void Editor::platformCopyFont()
{
}

void Editor::platformPasteFont()
{
}

template <typename PlatformDragData>
static RefPtr<DocumentFragment> createFragmentFromPlatformData(PlatformDragData& platformDragData, LocalFrame& frame)
{
    if (containsFilenames(&platformDragData)) {
        if (auto fragment = fragmentFromFilenames(frame.document(), &platformDragData))
            return fragment;
    }

    if (containsHTML(&platformDragData)) {
        if (RefPtr<DocumentFragment> fragment = fragmentFromHTML(frame.document(), &platformDragData))
            return fragment;
    }
    return nullptr;
}

RefPtr<DocumentFragment> Editor::webContentFromPasteboard(Pasteboard& pasteboard, const SimpleRange&, bool /*allowPlainText*/, bool& /*chosePlainText*/)
{
    if (COMPtr<IDataObject> platformDragData = pasteboard.dataObject())
        return createFragmentFromPlatformData(*platformDragData, *document().frame());

    return createFragmentFromPlatformData(pasteboard.dragDataMap(), *document().frame());
}

} // namespace WebCore
