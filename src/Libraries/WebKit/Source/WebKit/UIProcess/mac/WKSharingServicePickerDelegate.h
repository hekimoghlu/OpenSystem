/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
#import <wtf/text/WTFString.h>

namespace WebKit {
class WebContextMenuProxyMac;
}

@class NSSharingServicePicker;

@interface WKSharingServicePickerDelegate : NSObject <NSSharingServiceDelegate, NSSharingServicePickerDelegate> {
    WebKit::WebContextMenuProxyMac* _menuProxy;
    RetainPtr<NSSharingServicePicker> _picker;
    BOOL _filterEditingServices;
    BOOL _handleEditingReplacement;
    NSRect _sourceFrame;
    String _attachmentID;
}

+ (WKSharingServicePickerDelegate *)sharedSharingServicePickerDelegate;
- (WebKit::WebContextMenuProxyMac*)menuProxy;
- (void)setMenuProxy:(WebKit::WebContextMenuProxyMac*)menuProxy;
- (void)setPicker:(NSSharingServicePicker *)picker;
- (void)setFiltersEditingServices:(BOOL)filtersEditingServices;
- (void)setHandlesEditingReplacement:(BOOL)handlesEditingReplacement;
- (void)setSourceFrame:(NSRect)sourceFrame;
- (void)setAttachmentID:(String)attachmentID;
- (void)removeBackground;

@end

#endif // ENABLE(SERVICE_CONTROLS)
