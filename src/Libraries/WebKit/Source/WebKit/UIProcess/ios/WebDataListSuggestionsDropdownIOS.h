/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
#if PLATFORM(IOS_FAMILY)

#import "UIKitSPI.h"
#import "WKBrowserEngineDefinitions.h"
#import "WebDataListSuggestionsDropdown.h"
#import <pal/spi/ios/BrowserEngineKitSPI.h>
#import <wtf/RetainPtr.h>
#import <wtf/Vector.h>

OBJC_CLASS WKContentView;

@interface WKDataListTextSuggestion : WKBETextSuggestion
+ (instancetype)textSuggestionWithInputText:(NSString *)inputText;
@end

@interface WKDataListSuggestionsControl : NSObject

@property (nonatomic, readonly) BOOL isShowingSuggestions;

- (instancetype)initWithInformation:(WebCore::DataListSuggestionInformation&&)information inView:(WKContentView *)view;
- (void)updateWithInformation:(WebCore::DataListSuggestionInformation&&)information;
- (void)didSelectOptionAtIndex:(NSInteger)index;
- (void)invalidate;

@end

namespace WebKit {

class WebDataListSuggestionsDropdownIOS : public WebDataListSuggestionsDropdown {
public:
    static Ref<WebDataListSuggestionsDropdownIOS> create(WebPageProxy&, WKContentView *);

    void didSelectOption(const String&);

private:
    WebDataListSuggestionsDropdownIOS(WebPageProxy&, WKContentView *);

    void show(WebCore::DataListSuggestionInformation&&) final;
    void handleKeydownWithIdentifier(const String&) final;
    void close() final;

    WKContentView *m_contentView;
    RetainPtr<WKDataListSuggestionsControl> m_suggestionsControl;
};

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
