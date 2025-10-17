/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 19, 2023.
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
#import "WKQuickLookPreviewController.h"

#if HAVE(QUICKLOOK_PREVIEW_ITEM_DATA_PROVIDER)

#import "WebPageProxy.h"
#import <wtf/RetainPtr.h>
#import <wtf/URL.h>
#import <pal/mac/QuickLookUISoftLink.h>

@interface WKQuickLookPreviewController () <QLPreviewPanelDelegate, QLPreviewPanelDataSource>
@end

@implementation WKQuickLookPreviewController {
    RetainPtr<QLItem> _item;
    RetainPtr<NSData> _imageData;
    WebKit::QuickLookPreviewActivity _activity;
}

- (instancetype)initWithPage:(WebKit::WebPageProxy&)page imageData:(NSData *)imageData title:(NSString *)title imageURL:(NSURL *)imageURL activity:(WebKit::QuickLookPreviewActivity)activity
{
    if (!(self = [super init]))
        return nil;

    _activity = activity;
    _imageData = imageData;
    _item = adoptNS([PAL::allocQLItemInstance() initWithDataProvider:(id)self contentType:UTTypePNG previewTitle:title]);
    if ([_item respondsToSelector:@selector(setPreviewOptions:)]) {
        auto previewOptions = adoptNS([[NSMutableDictionary alloc] initWithCapacity:2]);
        if (imageURL)
            [previewOptions setObject:imageURL forKey:@"imageURL"];
        if (NSURL *pageURL = URL { page.currentURL() })
            [previewOptions setObject:pageURL forKey:@"pageURL"];
        [_item setPreviewOptions:previewOptions.get()];
    }

    return self;
}

- (void)beginControl:(QLPreviewPanel *)panel
{
    panel.dataSource = self;
    panel.delegate = self;
}

- (void)endControl:(QLPreviewPanel *)panel
{
    if (panel.dataSource == self)
        panel.dataSource = nil;

    if (panel.delegate == self)
        panel.delegate = nil;
}

- (void)closePanelIfNecessary
{
    if (!PAL::isQuickLookUIFrameworkAvailable() || ![PAL::getQLPreviewPanelClass() sharedPreviewPanelExists])
        return;

    if (auto panel = [PAL::getQLPreviewPanelClass() sharedPreviewPanel]; [self isControlling:panel])
        [panel close];
}

- (BOOL)isControlling:(QLPreviewPanel *)panel
{
    return panel.dataSource == self && panel.delegate == self;
}

#pragma mark - QLPreviewItemDataProvider

- (NSData *)provideDataForItem:(QLItem *)item
{
    ASSERT(item == _item);
    return _imageData.get();
}

#pragma mark - QLPreviewPanelDataSource

- (NSInteger)numberOfPreviewItemsInPreviewPanel:(QLPreviewPanel *)panel
{
    return 1;
}

- (id <QLPreviewItem>)previewPanel:(QLPreviewPanel *)panel previewItemAtIndex:(NSInteger)index
{
    ASSERT(!index);
    return _item.get();
}

#pragma mark - QLPreviewPanelDelegate

- (QLPreviewActivity)previewPanel:(QLPreviewPanel *)previewPanel initialActivityForItem:(id <QLPreviewItem>)item
{
    return _activity == WebKit::QuickLookPreviewActivity::VisualSearch ? QLPreviewActivityVisualSearch : QLPreviewActivityNone;
}

@end

#endif // HAVE(QUICKLOOK_PREVIEW_ITEM_DATA_PROVIDER)
