/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
#import "WKSharingServicePickerDelegate.h"

#if ENABLE(SERVICE_CONTROLS)

#import "APIAttachment.h"
#import "WKObject.h"
#import "WebContextMenuProxyMac.h"
#import "WebPageProxy.h"
#import "_WKAttachmentInternal.h"
#import <WebCore/LegacyNSPasteboardTypes.h>
#import <pal/spi/mac/NSSharingServicePickerSPI.h>
#import <pal/spi/mac/NSSharingServiceSPI.h>
#import <wtf/cocoa/SpanCocoa.h>
#import <wtf/text/WTFString.h>

// FIXME: We probably need to hang on the picker itself until the context menu operation is done, and this object will probably do that.
@implementation WKSharingServicePickerDelegate

+ (WKSharingServicePickerDelegate*)sharedSharingServicePickerDelegate
{
    static WKSharingServicePickerDelegate* delegate = [[WKSharingServicePickerDelegate alloc] init];
    return delegate;
}

- (WebKit::WebContextMenuProxyMac*)menuProxy
{
    return _menuProxy;
}

- (void)setMenuProxy:(WebKit::WebContextMenuProxyMac*)menuProxy
{
    _menuProxy = menuProxy;
}

- (void)setPicker:(NSSharingServicePicker *)picker
{
    _picker = picker;
}

- (void)setFiltersEditingServices:(BOOL)filtersEditingServices
{
    _filterEditingServices = filtersEditingServices;
}

- (void)setHandlesEditingReplacement:(BOOL)handlesEditingReplacement
{
    _handleEditingReplacement = handlesEditingReplacement;
}

- (void)setSourceFrame:(NSRect)sourceFrame
{
    _sourceFrame = sourceFrame;
}

- (void)setAttachmentID:(String)attachmentID
{
    _attachmentID = attachmentID;
}

- (NSArray *)sharingServicePicker:(NSSharingServicePicker *)sharingServicePicker sharingServicesForItems:(NSArray *)items mask:(NSSharingServiceMask)mask proposedSharingServices:(NSArray *)proposedServices
{
    if (!_filterEditingServices)
        return proposedServices;

    NSMutableArray *services = [NSMutableArray arrayWithCapacity:proposedServices.count];
    
    for (NSSharingService *service in proposedServices) {
        if (service.type != NSSharingServiceTypeEditor)
            [services addObject:service];
    }
    
    return services;
}

- (id <NSSharingServiceDelegate>)sharingServicePicker:(NSSharingServicePicker *)sharingServicePicker delegateForSharingService:(NSSharingService *)sharingService
{
    return self;
}

- (NSRect)sharingService:(NSSharingService *)sharingService sourceFrameOnScreenForShareItem:(id <NSPasteboardWriting>)item
{
    return _sourceFrame;
}

- (void)sharingService:(NSSharingService *)sharingService willShareItems:(NSArray *)items
{
    _menuProxy->clearServicesMenu();
}

- (void)sharingService:(NSSharingService *)sharingService didShareItems:(NSArray *)items
{
    // We only care about what item was shared if we were interested in editor services
    // (i.e., if we plan on replacing the selection or controlled image with the returned item)
    if (!_handleEditingReplacement)
        return;

    if (!items.count)
        return;

    Vector<String> types;
    std::span<const uint8_t> dataReference;

    id item = [items objectAtIndex:0];

    if ([item isKindOfClass:[NSAttributedString class]]) {
        NSData *data = [item RTFDFromRange:NSMakeRange(0, [item length]) documentAttributes:@{ }];
        dataReference = span(data);

        types.append(NSPasteboardTypeRTFD);
        types.append(WebCore::legacyRTFDPasteboardType());
    } else if ([item isKindOfClass:[NSData class]]) {
        NSData *data = (NSData *)item;
        RetainPtr<CGImageSourceRef> source = adoptCF(CGImageSourceCreateWithData((CFDataRef)data, NULL));
        RetainPtr<CGImageRef> image = adoptCF(CGImageSourceCreateImageAtIndex(source.get(), 0, NULL));

        if (!image)
            return;

        dataReference = span(data);
        types.append(NSPasteboardTypeTIFF);
    } else if ([item isKindOfClass:[NSItemProvider class]]) {
        NSItemProvider *itemProvider = (NSItemProvider *)item;
        
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        WeakPtr weakPage = _menuProxy->page();
        NSString *itemUTI = itemProvider.registeredTypeIdentifiers.firstObject;
        [itemProvider loadDataRepresentationForTypeIdentifier:itemUTI completionHandler:[weakPage, attachmentID = _attachmentID, itemUTI](NSData *data, NSError *error) {
            RefPtr webPage = weakPage.get();
            
            if (!webPage)
                return;
            
            if (error)
                return;
            
            auto apiAttachment = webPage->attachmentForIdentifier(attachmentID);
            if (!apiAttachment)
                return;
            
            auto attachment = wrapper(apiAttachment);
            [attachment setData:data newContentType:itemUTI];
            webPage->didInvalidateDataForAttachment(*apiAttachment.get());
        }];
ALLOW_DEPRECATED_DECLARATIONS_END
        return;
    } else {
        LOG_ERROR("sharingService:didShareItems: - Unknown item type returned\n");
        return;
    }

    // FIXME: We should adopt replaceSelectionWithAttributedString instead of bouncing through the (fake) pasteboard.
    _menuProxy->page()->replaceSelectionWithPasteboardData(types, dataReference);
}

- (NSWindow *)sharingService:(NSSharingService *)sharingService sourceWindowForShareItems:(NSArray *)items sharingContentScope:(NSSharingContentScope *)sharingContentScope
{
    return _menuProxy->window();
}

- (void)removeBackground
{
    _menuProxy->removeBackgroundFromControlledImage();
}

@end

#endif // ENABLE(SERVICE_CONTROLS)
