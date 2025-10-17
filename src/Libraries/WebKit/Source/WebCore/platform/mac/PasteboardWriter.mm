/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
#import "PasteboardWriter.h"

#if PLATFORM(MAC)

#import "LegacyNSPasteboardTypes.h"
#import "Pasteboard.h"
#import "PasteboardWriterData.h"
#import "SharedBuffer.h"
#import <pal/spi/mac/NSPasteboardSPI.h>
#import <wtf/cocoa/TypeCastsCocoa.h>

namespace WebCore {

static RetainPtr<NSString> toUTI(NSString *pasteboardType)
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return bridge_cast(adoptCF(UTTypeCreatePreferredIdentifierForTag(kUTTagClassNSPboardType, bridge_cast(pasteboardType), nullptr)));
ALLOW_DEPRECATED_DECLARATIONS_END
}

static RetainPtr<NSString> toUTIUnlessAlreadyUTI(NSString *type)
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    if (UTTypeIsDeclared(bridge_cast(type)) || UTTypeIsDynamic(bridge_cast(type))) {
        // This is already a UTI.
        return type;
    }
ALLOW_DEPRECATED_DECLARATIONS_END

    return toUTI(type);
}

RetainPtr<id <NSPasteboardWriting>> createPasteboardWriter(const PasteboardWriterData& data)
{
    auto pasteboardItem = adoptNS([[NSPasteboardItem alloc] init]);

    if (auto& plainText = data.plainText()) {
        [pasteboardItem setString:plainText->text forType:NSPasteboardTypeString];
        if (plainText->canSmartCopyOrDelete) {
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
            auto smartPasteType = bridge_cast(adoptCF(UTTypeCreatePreferredIdentifierForTag(kUTTagClassNSPboardType, bridge_cast(_NXSmartPaste), nullptr)));
ALLOW_DEPRECATED_DECLARATIONS_END
            [pasteboardItem setData:[NSData data] forType:smartPasteType.get()];
        }
    }

    if (auto& urlData = data.urlData()) {
        NSURL *cocoaURL = urlData->url;
        NSString *userVisibleString = urlData->userVisibleForm;
        NSString *title = (NSString *)urlData->title;
        if (!title.length) {
            title = cocoaURL.path.lastPathComponent;
            if (!title.length)
                title = userVisibleString;
        }

        // WebURLsWithTitlesPboardType.
        // FIXME: This could use StringView (the one that creates NSString) to save an allocation
        auto paths = adoptNS([[NSArray alloc] initWithObjects:@[ @[ cocoaURL.absoluteString ] ], @[ urlData->title.trim(deprecatedIsSpaceOrNewline) ], nil]);
        [pasteboardItem setPropertyList:paths.get() forType:toUTI(@"WebURLsWithTitlesPboardType").get()];

        // NSURLPboardType.
        if (NSURL *baseCocoaURL = cocoaURL.baseURL)
            [pasteboardItem setPropertyList:@[ cocoaURL.relativeString, baseCocoaURL.absoluteString ] forType:toUTI(WebCore::legacyURLPasteboardType()).get()];
        else if (cocoaURL)
            [pasteboardItem setPropertyList:@[ cocoaURL.absoluteString, @"" ] forType:toUTI(WebCore::legacyURLPasteboardType()).get()];
        else
            [pasteboardItem setPropertyList:@[ @"", @"" ] forType:toUTI(WebCore::legacyURLPasteboardType()).get()];

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        if (cocoaURL.fileURL)
            [pasteboardItem setString:cocoaURL.absoluteString forType:(NSString *)kUTTypeFileURL];
        [pasteboardItem setString:userVisibleString forType:(NSString *)kUTTypeURL];
ALLOW_DEPRECATED_DECLARATIONS_END

        // WebURLNamePboardType.
        [pasteboardItem setString:title forType:@"public.url-name"];

        // NSPasteboardTypeString.
        [pasteboardItem setString:userVisibleString forType:NSPasteboardTypeString];
    }

    if (auto& webContent = data.webContent()) {
        if (webContent->canSmartCopyOrDelete) {
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
            auto smartPasteType = bridge_cast(adoptCF(UTTypeCreatePreferredIdentifierForTag(kUTTagClassNSPboardType, bridge_cast(_NXSmartPaste), nullptr)));
ALLOW_DEPRECATED_DECLARATIONS_END
            [pasteboardItem setData:[NSData data] forType:smartPasteType.get()];
        }
        if (webContent->dataInWebArchiveFormat) {
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
            auto webArchiveType = bridge_cast(adoptCF(UTTypeCreatePreferredIdentifierForTag(kUTTagClassNSPboardType, CFSTR("Apple Web Archive pasteboard type"), nullptr)));
ALLOW_DEPRECATED_DECLARATIONS_END
            [pasteboardItem setData:webContent->dataInWebArchiveFormat->createNSData().get() forType:webArchiveType.get()];
        }
        if (webContent->dataInRTFDFormat)
            [pasteboardItem setData:webContent->dataInRTFDFormat->createNSData().get() forType:NSPasteboardTypeRTFD];
        if (webContent->dataInRTFFormat)
            [pasteboardItem setData:webContent->dataInRTFFormat->createNSData().get() forType:NSPasteboardTypeRTF];
        if (!webContent->dataInHTMLFormat.isNull())
            [pasteboardItem setString:webContent->dataInHTMLFormat forType:NSPasteboardTypeHTML];
        if (!webContent->dataInStringFormat.isNull())
            [pasteboardItem setString:webContent->dataInStringFormat forType:NSPasteboardTypeString];

        for (unsigned i = 0; i < webContent->clientTypesAndData.size(); ++i)
            [pasteboardItem setData:webContent->clientTypesAndData[i].second->createNSData().get() forType:toUTIUnlessAlreadyUTI(webContent->clientTypesAndData[i].first).get()];

        PasteboardCustomData customData;
        customData.setOrigin(webContent->contentOrigin);
        [pasteboardItem setData:customData.createSharedBuffer()->createNSData().get() forType:toUTIUnlessAlreadyUTI(String(PasteboardCustomData::cocoaType())).get()];
    }

    return pasteboardItem;
}

}

#endif
