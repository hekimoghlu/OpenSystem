/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
#import "WebSelectionServiceController.h"

#if ENABLE(SERVICE_CONTROLS)

#import "WebViewInternal.h"
#import <WebCore/FrameSelection.h>
#import <WebCore/HTMLConverter.h>
#import <WebCore/Range.h>
#import <WebCore/markup.h>
#import <pal/spi/mac/NSSharingServiceSPI.h>
#import <wtf/TZoneMallocInlines.h>

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebSelectionServiceController);

WebSelectionServiceController::WebSelectionServiceController(WebView *webView) 
    : WebSharingServicePickerClient(webView)
{
}

void WebSelectionServiceController::handleSelectionServiceClick(WebCore::FrameSelection& selection, const Vector<String>& /*telephoneNumbers*/, const WebCore::IntPoint& point)
{
    Page* page = [m_webView page];
    if (!page)
        return;

    auto range = selection.selection().firstRange();
    if (!range)
        return;

    auto attributedSelection = attributedString(*range, WebCore::IgnoreUserSelectNone::Yes).nsAttributedString();
    if (!attributedSelection)
        return;

    auto items = @[ attributedSelection.get() ];
    bool isEditable = selection.selection().isContentEditable();

    m_sharingServicePickerController = adoptNS([[WebSharingServicePickerController alloc] initWithItems:items includeEditorServices:isEditable client:this style:NSSharingServicePickerStyleTextSelection]);

    auto menu = adoptNS([[m_sharingServicePickerController menu] copy]);
    [menu setShowsStateColumn:YES];
    [menu popUpMenuPositioningItem:nil atLocation:[m_webView convertPoint:point toView:nil] inView:m_webView];
}

static bool hasCompatibleServicesForItems(NSArray *items)
{
    return [NSSharingService sharingServicesForItems:items mask:NSSharingServiceMaskViewer | NSSharingServiceMaskEditor].count;
}

bool WebSelectionServiceController::hasRelevantSelectionServices(bool isTextOnly) const
{
    RetainPtr<NSAttributedString> attributedString = adoptNS([[NSAttributedString alloc] initWithString:@"a"]);

    bool hasSelectionServices = hasCompatibleServicesForItems(@[ attributedString.get() ]);
    if (isTextOnly && hasSelectionServices)
        return true;

    auto attachment = adoptNS([[NSTextAttachment alloc] init]);
    auto image = adoptNS([[NSImage alloc] init]);
    auto cell = adoptNS([[NSTextAttachmentCell alloc] initImageCell:image.get()]);
    [attachment setAttachmentCell:cell.get()];
    NSMutableAttributedString *attributedStringWithRichContent = (NSMutableAttributedString *)[NSMutableAttributedString attributedStringWithAttachment:attachment.get()];
    [attributedStringWithRichContent appendAttributedString:attributedString.get()];

    return hasCompatibleServicesForItems(@[ attributedStringWithRichContent ]);
}

void WebSelectionServiceController::sharingServicePickerWillBeDestroyed(WebSharingServicePickerController &)
{
    m_sharingServicePickerController = nil;
}

#endif
