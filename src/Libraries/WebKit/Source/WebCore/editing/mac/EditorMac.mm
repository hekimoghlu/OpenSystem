/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 27, 2024.
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
#import "config.h"
#import "Editor.h"

#if PLATFORM(MAC)

#import "Blob.h"
#import "CSSPrimitiveValueMappings.h"
#import "CSSValuePool.h"
#import "Color.h"
#import "ColorCocoa.h"
#import "ColorSerialization.h"
#import "DataTransfer.h"
#import "DocumentFragment.h"
#import "Editing.h"
#import "EditorClient.h"
#import "FontAttributes.h"
#import "FontShadow.h"
#import "HTMLConverter.h"
#import "HTMLElement.h"
#import "HTMLNames.h"
#import "LegacyNSPasteboardTypes.h"
#import "LegacyWebArchive.h"
#import "LocalFrame.h"
#import "LocalFrameView.h"
#import "MutableStyleProperties.h"
#import "PagePasteboardContext.h"
#import "Pasteboard.h"
#import "PasteboardStrategy.h"
#import "PlatformStrategies.h"
#import "RenderBlock.h"
#import "RenderImage.h"
#import "SharedBuffer.h"
#import "WebContentReader.h"
#import "WebNSAttributedStringExtras.h"
#import "markup.h"
#import <AppKit/AppKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <wtf/RetainPtr.h>
#import <wtf/RuntimeApplicationChecks.h>
#import <wtf/cocoa/NSURLExtras.h>
#import <wtf/cocoa/TypeCastsCocoa.h>

namespace WebCore {

void Editor::pasteWithPasteboard(Pasteboard* pasteboard, OptionSet<PasteOption> options)
{
    auto range = selectedRange();

    // FIXME: How can this hard-coded pasteboard name be right, given that the passed-in pasteboard has a name?
    client()->setInsertionPasteboard(NSPasteboardNameGeneral);

    bool chosePlainText;
    RefPtr<DocumentFragment> fragment = webContentFromPasteboard(*pasteboard, *range, options.contains(PasteOption::AllowPlainText), chosePlainText);

    if (fragment && options.contains(PasteOption::AsQuotation))
        quoteFragmentForPasting(*fragment);

    if (fragment && shouldInsertFragment(*fragment, range, EditorInsertAction::Pasted))
        pasteAsFragment(fragment.releaseNonNull(), canSmartReplaceWithPasteboard(*pasteboard), false, options.contains(PasteOption::IgnoreMailBlockquote) ? MailBlockquoteHandling::IgnoreBlockquote : MailBlockquoteHandling::RespectBlockquote );
    
    if (auto* client = this->client())
        client->setInsertionPasteboard(String());
}

void Editor::platformCopyFont()
{
    Pasteboard pasteboard(PagePasteboardContext::create(document().pageID()), NSPasteboardNameFont);

    auto fontSampleString = adoptNS([[NSAttributedString alloc] initWithString:@"x" attributes:fontAttributesAtSelectionStart().createDictionary().get()]);
    auto fontData = RetainPtr([fontSampleString RTFFromRange:NSMakeRange(0, [fontSampleString length]) documentAttributes:@{ }]);

    PasteboardBuffer pasteboardBuffer;
    pasteboardBuffer.contentOrigin = document().originIdentifierForPasteboard();
    pasteboardBuffer.type = legacyFontPasteboardType();
    pasteboardBuffer.data = SharedBuffer::create(fontData.get());
    pasteboard.write(pasteboardBuffer);
}

void Editor::platformPasteFont()
{
    Pasteboard pasteboard(PagePasteboardContext::create(document().pageID()), NSPasteboardNameFont);

    client()->setInsertionPasteboard(pasteboard.name());

    RetainPtr<NSData> fontData;
    if (auto buffer = pasteboard.readBuffer(std::nullopt, legacyFontPasteboardType()))
        fontData = buffer->createNSData();
    auto fontSampleString = adoptNS([[NSAttributedString alloc] initWithRTF:fontData.get() documentAttributes:nil]);
    auto fontAttributes = RetainPtr([fontSampleString fontAttributesInRange:NSMakeRange(0, 1)]);

    auto style = MutableStyleProperties::create();

    Color backgroundColor;
    if (NSColor *nsBackgroundColor = dynamic_objc_cast<NSColor>([fontAttributes objectForKey:NSBackgroundColorAttributeName]))
        backgroundColor = colorFromCocoaColor(nsBackgroundColor);
    if (!backgroundColor.isValid())
        backgroundColor = Color::transparentBlack;
    style->setProperty(CSSPropertyBackgroundColor, CSSValuePool::singleton().createColorValue(backgroundColor));

    if (NSFont *font = dynamic_objc_cast<NSFont>([fontAttributes objectForKey:NSFontAttributeName])) {
        // FIXME: Need more sophisticated escaping code if we want to handle family names
        // with characters like single quote or backslash in their names.
        style->setProperty(CSSPropertyFontFamily, [NSString stringWithFormat:@"'%@'", [font familyName]]);
        style->setProperty(CSSPropertyFontSize, CSSPrimitiveValue::create([font pointSize], CSSUnitType::CSS_PX));
        // FIXME: Map to the entire range of CSS weight values.
        style->setProperty(CSSPropertyFontWeight, ([NSFontManager.sharedFontManager weightOfFont:font] >= 7) ? CSSValueBold : CSSValueNormal);
        style->setProperty(CSSPropertyFontStyle, ([NSFontManager.sharedFontManager traitsOfFont:font] & NSItalicFontMask) ? CSSValueItalic : CSSValueNormal);
    } else {
        style->setProperty(CSSPropertyFontFamily, "Helvetica"_s);
        style->setProperty(CSSPropertyFontSize, CSSPrimitiveValue::create(12, CSSUnitType::CSS_PX));
        style->setProperty(CSSPropertyFontWeight, CSSValueNormal);
        style->setProperty(CSSPropertyFontStyle, CSSValueNormal);
    }

    Color foregroundColor;
    if (NSColor *nsForegroundColor = dynamic_objc_cast<NSColor>([fontAttributes objectForKey:NSForegroundColorAttributeName])) {
        foregroundColor = colorFromCocoaColor(nsForegroundColor);
        if (!foregroundColor.isValid())
            foregroundColor = Color::transparentBlack;
    } else
        foregroundColor = Color::black;
    style->setProperty(CSSPropertyColor, CSSValuePool::singleton().createColorValue(foregroundColor));

    FontShadow fontShadow;
    if (NSShadow *nsFontShadow = dynamic_objc_cast<NSShadow>([fontAttributes objectForKey:NSShadowAttributeName]))
        fontShadow = fontShadowFromNSShadow(nsFontShadow);
    style->setProperty(CSSPropertyTextShadow, serializationForCSS(fontShadow));

    auto superscriptStyle = [[fontAttributes objectForKey:NSSuperscriptAttributeName] intValue];
    style->setProperty(CSSPropertyVerticalAlign, (superscriptStyle > 0) ? CSSValueSuper : ((superscriptStyle < 0) ? CSSValueSub : CSSValueBaseline));

    // FIXME: Underline wins here if we have both (see bug 3790443).
    auto underlineStyle = [[fontAttributes objectForKey:NSUnderlineStyleAttributeName] intValue];
    auto strikethroughStyle = [[fontAttributes objectForKey:NSStrikethroughStyleAttributeName] intValue];
    style->setProperty(CSSPropertyWebkitTextDecorationsInEffect, (underlineStyle != NSUnderlineStyleNone) ? CSSValueUnderline : ((strikethroughStyle != NSUnderlineStyleNone) ? CSSValueLineThrough : CSSValueNone));

    applyStyleToSelection(style.ptr(), EditAction::PasteFont);

    client()->setInsertionPasteboard(String());
}

RefPtr<SharedBuffer> Editor::imageInWebArchiveFormat(Element& imageElement)
{
    auto archive = LegacyWebArchive::create(imageElement);
    if (!archive)
        return nullptr;
    return SharedBuffer::create(archive->rawDataRepresentation().get());
}

RefPtr<SharedBuffer> Editor::dataSelectionForPasteboard(const String& pasteboardType)
{
    // FIXME: The interface to this function is awkward. We'd probably be better off with three separate functions. As of this writing, this is only used in WebKit2 to implement the method -[WKView writeSelectionToPasteboard:types:], which is only used to support OS X services.

    // FIXME: Does this function really need to use adjustedSelectionRange()? Because writeSelectionToPasteboard() just uses selectedRange(). This is the only function in WebKit that uses adjustedSelectionRange.
    if (!canCopy())
        return nullptr;

    if (pasteboardType == WebArchivePboardType || pasteboardType == String(UTTypeWebArchive.identifier))
        return selectionInWebArchiveFormat();

    if (pasteboardType == String(legacyRTFDPasteboardType()))
        return dataInRTFDFormat(attributedString(*adjustedSelectionRange(), IgnoreUserSelectNone::Yes).nsAttributedString().get());

    if (pasteboardType == String(legacyRTFPasteboardType())) {
        auto string = attributedString(*adjustedSelectionRange(), IgnoreUserSelectNone::Yes).nsAttributedString();
        // FIXME: Why is this stripping needed here, but not in writeSelectionToPasteboard?
        if ([string containsAttachments])
            string = attributedStringByStrippingAttachmentCharacters(string.get());
        return dataInRTFFormat(string.get());
    }

    return nullptr;
}

static void getImage(Element& imageElement, RefPtr<Image>& image, CachedImage*& cachedImage)
{
    CheckedPtr renderImage = dynamicDowncast<RenderImage>(imageElement.renderer());
    if (!renderImage)
        return;

    CachedResourceHandle tentativeCachedImage = renderImage->cachedImage();
    if (!tentativeCachedImage || tentativeCachedImage->errorOccurred())
        return;

    image = tentativeCachedImage->imageForRenderer(renderImage.get());
    if (!image)
        return;

    cachedImage = tentativeCachedImage.get();
}

void Editor::selectionWillChange()
{
    if (!hasComposition() || ignoreSelectionChanges() || document().selection().isNone() || !document().hasLivingRenderTree())
        return;

    cancelComposition();
    client()->canceledComposition();
}

String Editor::plainTextFromPasteboard(const PasteboardPlainText& text)
{
    auto string = text.text;

    // FIXME: It's not clear this is 100% correct since we know -[NSURL URLWithString:] does not handle
    // all the same cases we handle well in the URL code for creating an NSURL.
    if (text.isURL)
        string = WTF::userVisibleString([NSURL URLWithString:string]);

    // FIXME: WTF should offer a non-Mac-specific way to convert string to precomposed form so we can do it for all platforms.
    return [(NSString *)string precomposedStringWithCanonicalMapping];
}

void Editor::writeImageToPasteboard(Pasteboard& pasteboard, Element& imageElement, const URL& url, const String& title)
{
    PasteboardImage pasteboardImage;

    CachedImage* cachedImage = nullptr;
    getImage(imageElement, pasteboardImage.image, cachedImage);
    if (!pasteboardImage.image)
        return;
    ASSERT(cachedImage);

    if (!pasteboard.isStatic())
        pasteboardImage.dataInWebArchiveFormat = imageInWebArchiveFormat(imageElement);

    if (auto imageRange = makeRangeSelectingNode(imageElement))
        pasteboardImage.dataInHTMLFormat = serializePreservingVisualAppearance(VisibleSelection { *imageRange }, ResolveURLs::YesExcludingURLsForPrivacy, SerializeComposedTree::Yes, IgnoreUserSelectNone::Yes);

    pasteboardImage.url.url = url;
    pasteboardImage.url.title = title;
    pasteboardImage.url.userVisibleForm = WTF::userVisibleString(pasteboardImage.url.url);
    if (auto* buffer = cachedImage->resourceBuffer())
        pasteboardImage.resourceData = buffer->makeContiguous();
    pasteboardImage.resourceMIMEType = cachedImage->response().mimeType();

    pasteboard.write(pasteboardImage);
}

bool Editor::writingSuggestionsSupportsSuffix()
{
    return true;
}

} // namespace WebCore

#endif // PLATFORM(MAC)
