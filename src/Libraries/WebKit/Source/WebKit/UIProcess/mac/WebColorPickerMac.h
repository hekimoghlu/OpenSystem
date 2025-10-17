/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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
#if USE(APPKIT)

#import "ColorControlSupportsAlpha.h"
#import "WebColorPicker.h"
#import <WebCore/IntRect.h>
#import <wtf/RetainPtr.h>
#import <wtf/Vector.h>

namespace WebCore {
class Color;
}

namespace WebKit {
class WebColorPickerMac;
}

@protocol WKColorPickerUIMac <NSObject>
- (void)setAndShowPicker:(WebKit::WebColorPickerMac*)picker withColor:(NSColor *)color supportsAlpha:(WebKit::ColorControlSupportsAlpha)supportsAlpha suggestions:(Vector<WebCore::Color>&&)suggestions;
- (void)invalidate;
- (void)setColor:(NSColor *)color;
- (void)didChooseColor:(id)sender;
@end

namespace WebKit {
    
class WebColorPickerMac final : public WebColorPicker {
public:        
    static Ref<WebColorPickerMac> create(WebColorPicker::Client*, const WebCore::Color&, const WebCore::IntRect&, WebKit::ColorControlSupportsAlpha, Vector<WebCore::Color>&&, NSView *);
    virtual ~WebColorPickerMac();

    void endPicker() final;
    void setSelectedColor(const WebCore::Color&) final;
    void showColorPicker(const WebCore::Color&) final;
    
    void didChooseColor(const WebCore::Color&);

private:
    WebColorPickerMac(WebColorPicker::Client*, const WebCore::Color&, const WebCore::IntRect&, WebKit::ColorControlSupportsAlpha, Vector<WebCore::Color>&&, NSView *);
    RetainPtr<NSObject<WKColorPickerUIMac> > m_colorPickerUI;
    WebKit::ColorControlSupportsAlpha m_supportsAlpha { WebKit::ColorControlSupportsAlpha::No };
    Vector<WebCore::Color> m_suggestions;
};

} // namespace WebKit

#endif // USE(APPKIT)
