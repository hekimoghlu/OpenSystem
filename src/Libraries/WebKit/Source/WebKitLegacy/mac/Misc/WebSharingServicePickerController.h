/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
#if ENABLE(SERVICE_CONTROLS)

#import <wtf/RetainPtr.h>

#if PLATFORM(MAC)
#import <pal/spi/mac/NSSharingServicePickerSPI.h>
#import <pal/spi/mac/NSSharingServiceSPI.h>
#endif

@class WebSharingServicePickerController;
@class WebView;

namespace WebCore {
class FloatRect;
class Page;
}

class WebContextMenuClient;

class WebSharingServicePickerClient {
public:
    virtual ~WebSharingServicePickerClient() { }

    virtual void sharingServicePickerWillBeDestroyed(WebSharingServicePickerController &);
    virtual WebCore::Page* pageForSharingServicePicker(WebSharingServicePickerController &);
    virtual RetainPtr<NSWindow> windowForSharingServicePicker(WebSharingServicePickerController &);

    virtual WebCore::FloatRect screenRectForCurrentSharingServicePickerItem(WebSharingServicePickerController &);
    virtual RetainPtr<NSImage> imageForCurrentSharingServicePickerItem(WebSharingServicePickerController &);

    WebView *webView() { return m_webView; }

protected:
    explicit WebSharingServicePickerClient(WebView *);
    WebView *m_webView;
};

@interface WebSharingServicePickerController : NSObject <NSSharingServiceDelegate, NSSharingServicePickerDelegate> {
    WebSharingServicePickerClient* _pickerClient;
    RetainPtr<NSSharingServicePicker> _picker;
    BOOL _includeEditorServices;
    BOOL _handleEditingReplacement;
}

- (instancetype)initWithItems:(NSArray *)items includeEditorServices:(BOOL)includeEditorServices client:(WebSharingServicePickerClient*)pickerClient style:(NSSharingServicePickerStyle)style;
- (instancetype)initWithSharingServicePicker:(NSSharingServicePicker *)sharingServicePicker client:(WebSharingServicePickerClient&)pickerClient;
- (NSMenu *)menu;
- (void)didShareImageData:(NSData *)data confirmDataIsValidTIFFData:(BOOL)confirmData;
- (void)clear;

@end

#endif
