/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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

#include "CachedImage.h"
#include "DocumentFragment.h"
#include "ElementInlines.h"
#include "HTMLEmbedElement.h"
#include "HTMLImageElement.h"
#include "HTMLInputElement.h"
#include "HTMLNames.h"
#include "HTMLObjectElement.h"
#include "LocalFrame.h"
#include "Pasteboard.h"
#include "RenderImage.h"
#include "SVGElementTypeHelpers.h"
#include "SVGImageElement.h"
#include "SelectionData.h"
#include "Settings.h"
#include "WebContentReader.h"
#include "XLinkNames.h"
#include "markup.h"

namespace WebCore {

void Editor::pasteWithPasteboard(Pasteboard* pasteboard, OptionSet<PasteOption> options)
{
    auto range = selectedRange();
    if (!range)
        return;

    bool chosePlainText;
    auto fragment = webContentFromPasteboard(*pasteboard, *range, options.contains(PasteOption::AllowPlainText), chosePlainText);

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

static const AtomString& elementURL(Element& element)
{
    if (is<HTMLImageElement>(element) || is<HTMLInputElement>(element))
        return element.attributeWithoutSynchronization(HTMLNames::srcAttr);
    if (is<SVGImageElement>(element))
        return element.attributeWithoutSynchronization(XLinkNames::hrefAttr);
    if (is<HTMLEmbedElement>(element) || is<HTMLObjectElement>(element))
        return element.imageSourceURL();
    return nullAtom();
}

static bool getImageForElement(Element& element, RefPtr<Image>& image)
{
    auto* renderer = element.renderer();
    if (!is<RenderImage>(renderer))
        return false;

    CachedImage* cachedImage = downcast<RenderImage>(*renderer).cachedImage();
    if (!cachedImage || cachedImage->errorOccurred())
        return false;

    image = cachedImage->imageForRenderer(renderer);
    return image;
}

void Editor::writeImageToPasteboard(Pasteboard& pasteboard, Element& imageElement, const URL&, const String& title)
{
    PasteboardImage pasteboardImage;

    if (!getImageForElement(imageElement, pasteboardImage.image))
        return;
    ASSERT(pasteboardImage.image);

    pasteboardImage.url.url = imageElement.document().completeURL(elementURL(imageElement));
    pasteboardImage.url.title = title;
    pasteboardImage.url.markup = serializeFragment(imageElement, SerializedNodes::SubtreeIncludingNode, nullptr, ResolveURLs::Yes);
    pasteboard.write(pasteboardImage);
}

void Editor::writeSelectionToPasteboard(Pasteboard& pasteboard)
{
    PasteboardWebContent pasteboardContent;
    pasteboardContent.canSmartCopyOrDelete = canSmartCopyOrDelete();
    pasteboardContent.text = selectedTextForDataTransfer();
    pasteboardContent.markup = serializePreservingVisualAppearance(document().selection().selection(), ResolveURLs::YesExcludingURLsForPrivacy, SerializeComposedTree::Yes, IgnoreUserSelectNone::Yes);
    pasteboardContent.contentOrigin = document().originIdentifierForPasteboard();
    pasteboard.write(pasteboardContent);
}

RefPtr<DocumentFragment> Editor::webContentFromPasteboard(Pasteboard& pasteboard, const SimpleRange& context, bool allowPlainText, bool& chosePlainText)
{
    WebContentReader reader(*document().frame(), context, allowPlainText);
    pasteboard.read(reader);
    chosePlainText = reader.madeFragmentFromPlainText();
    return reader.takeFragment();
}

} // namespace WebCore
