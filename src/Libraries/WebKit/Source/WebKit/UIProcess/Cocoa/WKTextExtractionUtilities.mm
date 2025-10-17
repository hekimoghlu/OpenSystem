/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 8, 2024.
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
#import "WKTextExtractionUtilities.h"

#import "WKTextExtractionItem.h"
#import <WebCore/TextExtraction.h>
#import <wtf/cocoa/VectorCocoa.h>

#import "WebKitSwiftSoftLink.h"

namespace WebKit {
using namespace WebCore;

void prepareTextExtractionSupportIfNeeded()
{
    // Preemptively soft link libWebKitSwift if it exists, so that the corresponding Swift extension
    // on WKWebView will be loaded.
    WebKitSwiftLibrary(true);
}

inline static WKTextExtractionContainer containerType(TextExtraction::ContainerType type)
{
    switch (type) {
    case TextExtraction::ContainerType::Root:
        return WKTextExtractionContainerRoot;
    case TextExtraction::ContainerType::ViewportConstrained:
        return WKTextExtractionContainerViewportConstrained;
    case TextExtraction::ContainerType::List:
        return WKTextExtractionContainerList;
    case TextExtraction::ContainerType::ListItem:
        return WKTextExtractionContainerListItem;
    case TextExtraction::ContainerType::BlockQuote:
        return WKTextExtractionContainerBlockQuote;
    case TextExtraction::ContainerType::Article:
        return WKTextExtractionContainerArticle;
    case TextExtraction::ContainerType::Section:
        return WKTextExtractionContainerSection;
    case TextExtraction::ContainerType::Nav:
        return WKTextExtractionContainerNav;
    case TextExtraction::ContainerType::Button:
        return WKTextExtractionContainerButton;
    }
}

inline static RetainPtr<WKTextExtractionTextItem> createWKTextItem(const TextExtraction::TextItemData& data, CGRect rectInWebView, NSArray<WKTextExtractionItem *> *children)
{
    RetainPtr<WKTextExtractionEditable> editable;
    if (data.editable) {
        editable = adoptNS([allocWKTextExtractionEditableInstance()
            initWithLabel:data.editable->label
            placeholder:data.editable->placeholder
            isSecure:static_cast<BOOL>(data.editable->isSecure)
            isFocused:static_cast<BOOL>(data.editable->isFocused)]);
    }

    auto selectedRange = NSMakeRange(NSNotFound, 0);
    if (auto range = data.selectedRange) {
        if (LIKELY(range->location + range->length <= data.content.length()))
            selectedRange = NSMakeRange(range->location, range->length);
    }

    auto links = createNSArray(data.links, [&](auto& linkAndRange) -> RetainPtr<WKTextExtractionLink> {
        auto& [url, range] = linkAndRange;
        if (UNLIKELY(range.location + range.length > data.content.length()))
            return { };
        return adoptNS([allocWKTextExtractionLinkInstance() initWithURL:url range:NSMakeRange(range.location, range.length)]);
    });

    return adoptNS([allocWKTextExtractionTextItemInstance()
        initWithContent:data.content
        selectedRange:selectedRange
        links:links.get()
        editable:editable.get()
        rectInWebView:rectInWebView
        children:children]);
}

inline static RetainPtr<WKTextExtractionItem> createItemWithChildren(const TextExtraction::Item& item, const RootViewToWebViewConverter& converter, NSArray<WKTextExtractionItem *> *children)
{
    auto rectInWebView = converter(item.rectInRootView);
    return WTF::switchOn(item.data,
        [&](const TextExtraction::TextItemData& data) -> RetainPtr<WKTextExtractionItem> {
            return createWKTextItem(data, rectInWebView, children);
        }, [&](const TextExtraction::ScrollableItemData& data) -> RetainPtr<WKTextExtractionItem> {
            return adoptNS([allocWKTextExtractionScrollableItemInstance() initWithContentSize:data.contentSize rectInWebView:rectInWebView children:children]);
        }, [&](const TextExtraction::ImageItemData& data) -> RetainPtr<WKTextExtractionItem> {
            return adoptNS([allocWKTextExtractionImageItemInstance() initWithName:data.name altText:data.altText rectInWebView:rectInWebView children:children]);
        }, [&](TextExtraction::ContainerType type) -> RetainPtr<WKTextExtractionItem> {
            return adoptNS([allocWKTextExtractionContainerItemInstance() initWithContainer:containerType(type) rectInWebView:rectInWebView children:children]);
        }
    );
}

static RetainPtr<WKTextExtractionItem> createItemRecursive(const TextExtraction::Item& item, const RootViewToWebViewConverter& converter)
{
    return createItemWithChildren(item, converter, createNSArray(item.children, [&](auto& child) {
        return createItemRecursive(child, converter);
    }).get());
}

RetainPtr<WKTextExtractionItem> createItem(const TextExtraction::Item& item, RootViewToWebViewConverter&& converter)
{
    if (!std::holds_alternative<TextExtraction::ContainerType>(item.data)) {
        ASSERT_NOT_REACHED();
        return nil;
    }

    if (std::get<TextExtraction::ContainerType>(item.data) != TextExtraction::ContainerType::Root) {
        ASSERT_NOT_REACHED();
        return nil;
    }

    return createItemRecursive(item, WTFMove(converter));
}

} // namespace WebKit
