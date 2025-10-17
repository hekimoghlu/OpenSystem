/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
#ifndef WebSelectionServiceController_h
#define WebSelectionServiceController_h

#if ENABLE(SERVICE_CONTROLS)

#import "WebSharingServicePickerController.h"
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/Vector.h>
#import <wtf/text/WTFString.h>

OBJC_CLASS NSImage;
OBJC_CLASS NSWindow;
OBJC_CLASS WebView;

namespace WebCore {
class FrameSelection;
class IntPoint;
}

class WebSelectionServiceController : public WebSharingServicePickerClient {
    WTF_MAKE_TZONE_ALLOCATED(WebSelectionServiceController);
public:
    WebSelectionServiceController(WebView*);

    void handleSelectionServiceClick(WebCore::FrameSelection&, const Vector<String>& telephoneNumbers, const WebCore::IntPoint&);
    bool hasRelevantSelectionServices(bool isTextOnly) const;

    // WebSharingServicePickerClient
    void sharingServicePickerWillBeDestroyed(WebSharingServicePickerController &) override;

private:
    RetainPtr<WebSharingServicePickerController> m_sharingServicePickerController;
};

#endif // ENABLE(SERVICE_CONTROLS)

#endif // WebSelectionServiceController_h
