/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
#import "DocumentEditingContext.h"

#if PLATFORM(IOS_FAMILY)

#import "UIKitSPI.h"
#import <WebCore/ElementContext.h>
#import <pal/spi/ios/BrowserEngineKitSPI.h>
#import <wtf/cocoa/VectorCocoa.h>

namespace WebKit {

static inline NSRange toNSRange(DocumentEditingContext::Range range)
{
    return NSMakeRange(range.location, range.length);
}

#if HAVE(UI_WK_DOCUMENT_CONTEXT)

template <typename ContextType>
void setOptionalEditingContextProperties(const DocumentEditingContext& context, ContextType *platformContext, OptionSet<DocumentEditingContextRequest::Options> options)
{
    for (auto& rect : context.textRects)
        [platformContext addTextRect:rect.rect forCharacterRange:toNSRange(rect.range)];

    [platformContext setAnnotatedText:context.annotatedText.nsAttributedString().get()];

#if HAVE(AUTOCORRECTION_ENHANCEMENTS)
    if (options.contains(DocumentEditingContextRequest::Options::AutocorrectedRanges)) {
        auto ranges = createNSArray(context.autocorrectedRanges, [&] (DocumentEditingContext::Range range) {
            return [NSValue valueWithRange:toNSRange(range)];
        });

        if ([platformContext respondsToSelector:@selector(setAutocorrectedRanges:)])
            [platformContext setAutocorrectedRanges:ranges.get()];
    }
#endif // HAVE(AUTOCORRECTION_ENHANCEMENTS)
}

#endif // HAVE(UI_WK_DOCUMENT_CONTEXT)

UIWKDocumentContext *DocumentEditingContext::toLegacyPlatformContext(OptionSet<DocumentEditingContextRequest::Options> options)
{
#if HAVE(UI_WK_DOCUMENT_CONTEXT)
    auto platformContext = adoptNS([[UIWKDocumentContext alloc] init]);

    if (options.contains(DocumentEditingContextRequest::Options::AttributedText)) {
        [platformContext setContextBefore:contextBefore.nsAttributedString().get()];
        [platformContext setSelectedText:selectedText.nsAttributedString().get()];
        [platformContext setContextAfter:contextAfter.nsAttributedString().get()];
        [platformContext setMarkedText:markedText.nsAttributedString().get()];
    } else if (options.contains(DocumentEditingContextRequest::Options::Text)) {
        [platformContext setContextBefore:[contextBefore.nsAttributedString() string]];
        [platformContext setSelectedText:[selectedText.nsAttributedString() string]];
        [platformContext setContextAfter:[contextAfter.nsAttributedString() string]];
        [platformContext setMarkedText:[markedText.nsAttributedString() string]];
    }

    [platformContext setSelectedRangeInMarkedText:toNSRange(selectedRangeInMarkedText)];
    setOptionalEditingContextProperties(*this, platformContext.get(), options);

    return platformContext.autorelease();
#else
    UNUSED_PARAM(options);
    return nil;
#endif
}

WKBETextDocumentContext *DocumentEditingContext::toPlatformContext(OptionSet<DocumentEditingContextRequest::Options> options)
{
#if HAVE(UI_WK_DOCUMENT_CONTEXT)
#if USE(BROWSERENGINEKIT)
    RetainPtr<WKBETextDocumentContext> platformContext;
    if (options.contains(DocumentEditingContextRequest::Options::AttributedText)) {
        platformContext = adoptNS([[WKBETextDocumentContext alloc] initWithAttributedSelectedText:selectedText.nsAttributedString().get()
            contextBefore:contextBefore.nsAttributedString().get()
            contextAfter:contextAfter.nsAttributedString().get()
            markedText:markedText.nsAttributedString().get()
            selectedRangeInMarkedText:toNSRange(selectedRangeInMarkedText)]);
    } else if (options.contains(DocumentEditingContextRequest::Options::Text)) {
        platformContext = adoptNS([[WKBETextDocumentContext alloc] initWithSelectedText:[selectedText.nsAttributedString() string]
            contextBefore:[contextBefore.nsAttributedString() string]
            contextAfter:[contextAfter.nsAttributedString() string]
            markedText:[markedText.nsAttributedString() string]
            selectedRangeInMarkedText:toNSRange(selectedRangeInMarkedText)]);
    } else
        ASSERT_NOT_REACHED_WITH_MESSAGE("%s expected at least Options::AttributedText or Options::Text", __PRETTY_FUNCTION__);
    setOptionalEditingContextProperties(*this, platformContext.get(), options);
    return platformContext.autorelease();
#else
    return toLegacyPlatformContext(options);
#endif
#else
    UNUSED_PARAM(options);
    return nil;
#endif
}

}

#endif
