/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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
#import "WebSharingServicePickerController.h"

#if ENABLE(SERVICE_CONTROLS)

#import "WebContextMenuClient.h"
#import "WebViewInternal.h"
#import <WebCore/BitmapImage.h>
#import <WebCore/ContextMenuController.h>
#import <WebCore/Document.h>
#import <WebCore/Editor.h>
#import <WebCore/FocusController.h>
#import <WebCore/FrameSelection.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/Page.h>

static NSString *serviceControlsPasteboardName = @"WebKitServiceControlsPasteboard";

WebSharingServicePickerClient::WebSharingServicePickerClient(WebView *webView)
    : m_webView(webView)
{
}

void WebSharingServicePickerClient::sharingServicePickerWillBeDestroyed(WebSharingServicePickerController &)
{
}

WebCore::Page* WebSharingServicePickerClient::pageForSharingServicePicker(WebSharingServicePickerController &)
{
    return [m_webView page];
}

RetainPtr<NSWindow> WebSharingServicePickerClient::windowForSharingServicePicker(WebSharingServicePickerController &)
{
    return [m_webView window];
}

WebCore::FloatRect WebSharingServicePickerClient::screenRectForCurrentSharingServicePickerItem(WebSharingServicePickerController &)
{
    return WebCore::FloatRect();
}

RetainPtr<NSImage> WebSharingServicePickerClient::imageForCurrentSharingServicePickerItem(WebSharingServicePickerController &)
{
    return nil;
}

@implementation WebSharingServicePickerController

#if ENABLE(SERVICE_CONTROLS)
- (instancetype)initWithItems:(NSArray *)items includeEditorServices:(BOOL)includeEditorServices client:(WebSharingServicePickerClient*)pickerClient style:(NSSharingServicePickerStyle)style
{
    if (!(self = [super init]))
        return nil;

    _picker = adoptNS([[NSSharingServicePicker alloc] initWithItems:items]);
    [_picker setStyle:style];
    [_picker setDelegate:self];

    _includeEditorServices = includeEditorServices;
    _handleEditingReplacement = includeEditorServices;
    _pickerClient = pickerClient;

    return self;
}

- (instancetype)initWithSharingServicePicker:(NSSharingServicePicker *)sharingServicePicker client:(WebSharingServicePickerClient&)pickerClient
{
    if (!(self = [super init]))
        return nil;

    _picker = sharingServicePicker;
    [_picker setDelegate:self];

    _includeEditorServices = YES;
    _pickerClient = &pickerClient;

    return self;
}
#endif // ENABLE(SERVICE_CONTROLS)


- (void)clear
{
    // Protect self from being dealloc'ed partway through this method.
    RetainPtr<WebSharingServicePickerController> protector(self);

    if (_pickerClient)
        _pickerClient->sharingServicePickerWillBeDestroyed(*self);

    _picker = nullptr;
    _pickerClient = nullptr;
}

- (NSMenu *)menu
{
    return [_picker menu];
}

- (void)didShareImageData:(NSData *)data confirmDataIsValidTIFFData:(BOOL)confirmData
{
    auto* page = _pickerClient->pageForSharingServicePicker(*self);
    if (!page)
        return;

    if (confirmData) {
        RetainPtr<NSImage> nsImage = adoptNS([[NSImage alloc] initWithData:data]);
        if (!nsImage) {
            LOG_ERROR("Shared image data cannot create a valid NSImage");
            return;
        }

        data = [nsImage TIFFRepresentation];
    }

    NSPasteboard *pasteboard = [NSPasteboard pasteboardWithName:serviceControlsPasteboardName];
    [pasteboard declareTypes:@[ NSPasteboardTypeTIFF ] owner:nil];
    [pasteboard setData:data forType:NSPasteboardTypeTIFF];

    if (RefPtr node = page->contextMenuController().context().hitTestResult().innerNode()) {
        if (RefPtr frame = node->document().frame())
            frame->editor().replaceNodeFromPasteboard(*node, serviceControlsPasteboardName);
    }

    [self clear];
}

#pragma mark NSSharingServicePickerDelegate methods

- (NSArray *)sharingServicePicker:(NSSharingServicePicker *)sharingServicePicker sharingServicesForItems:(NSArray *)items mask:(NSSharingServiceMask)mask proposedSharingServices:(NSArray *)proposedServices
{
    if (_includeEditorServices)
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

- (void)sharingServicePicker:(NSSharingServicePicker *)sharingServicePicker didChooseSharingService:(NSSharingService *)service
{
    if (!service)
        _pickerClient->sharingServicePickerWillBeDestroyed(*self);
}

#pragma mark NSSharingServiceDelegate methods

- (void)sharingService:(NSSharingService *)sharingService didShareItems:(NSArray *)items
{
    if (!_handleEditingReplacement)
        return;

    // We only send one item, so we should only get one item back.
    if ([items count] != 1)
        return;

    id item = [items objectAtIndex:0];

    if ([item isKindOfClass:[NSImage class]])
        [self didShareImageData:[item TIFFRepresentation] confirmDataIsValidTIFFData:NO];
    else if ([item isKindOfClass:[NSItemProvider class]]) {
        NSItemProvider *itemProvider = (NSItemProvider *)item;
        NSString *itemUTI = itemProvider.registeredTypeIdentifiers.firstObject;
        
        [itemProvider loadItemForTypeIdentifier:itemUTI options:nil completionHandler:^(id receivedData, NSError *dataError) {
            if (!receivedData) {
                LOG_ERROR("Did not receive data from NSItemProvider");
                return;
            }

            if (![receivedData isKindOfClass:[NSData class]]) {
                LOG_ERROR("Data received from NSItemProvider is not of type NSData");
                return;
            }

            [[NSOperationQueue mainQueue] addOperationWithBlock:^{
                [self didShareImageData:receivedData confirmDataIsValidTIFFData:YES];
            }];

        }];
    }
    else if ([item isKindOfClass:[NSAttributedString class]]) {
        if (RefPtr frame = _pickerClient->pageForSharingServicePicker(*self)->focusController().focusedOrMainFrame())
            frame->editor().replaceSelectionWithAttributedString(item);
    } else
        LOG_ERROR("sharingService:didShareItems: - Unknown item type returned\n");
}

- (void)sharingService:(NSSharingService *)sharingService didFailToShareItems:(NSArray *)items error:(NSError *)error
{
    [self clear];
}

- (NSRect)sharingService:(NSSharingService *)sharingService sourceFrameOnScreenForShareItem:(id <NSPasteboardWriting>)item
{
    if (!_pickerClient)
        return NSZeroRect;

    return _pickerClient->screenRectForCurrentSharingServicePickerItem(*self);
}

- (NSImage *)sharingService:(NSSharingService *)sharingService transitionImageForShareItem:(id <NSPasteboardWriting>)item contentRect:(NSRect *)contentRect
{
    if (!_pickerClient)
        return nil;

    return _pickerClient->imageForCurrentSharingServicePickerItem(*self).get();
}

- (NSWindow *)sharingService:(NSSharingService *)sharingService sourceWindowForShareItems:(NSArray *)items sharingContentScope:(NSSharingContentScope *)sharingContentScope
{
    return _pickerClient->windowForSharingServicePicker(*self).get();
}

@end

#endif // ENABLE(SERVICE_CONTROLS)
