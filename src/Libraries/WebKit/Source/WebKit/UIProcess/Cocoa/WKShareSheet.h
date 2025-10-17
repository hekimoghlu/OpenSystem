/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 19, 2022.
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
#import "WKAPICast.h"

#if HAVE(SHARE_SHEET_UI)

@class WKWebView;
@protocol WKShareSheetDelegate;

namespace WebCore {
struct ShareDataWithParsedURL;
}

namespace WebKit {
enum class PickerDismissalReason : uint8_t;
}

@interface WKShareSheet : NSObject

- (instancetype)initWithView:(WKWebView *)view;

- (void)presentWithParameters:(const WebCore::ShareDataWithParsedURL&)data inRect:(std::optional<WebCore::FloatRect>)rect completionHandler:(WTF::CompletionHandler<void(bool)>&&)completionHandler;

- (BOOL)dismissIfNeededWithReason:(WebKit::PickerDismissalReason)reason;

@property (nonatomic, weak) id <WKShareSheetDelegate> delegate;
@end

@protocol WKShareSheetDelegate <NSObject>
@optional
- (void)shareSheetDidDismiss:(WKShareSheet *)shareSheet;
- (void)shareSheet:(WKShareSheet *)shareSheet willShowActivityItems:(NSArray *)activityItems;
@end

#endif // HAVE(SHARE_SHEET_UI)
