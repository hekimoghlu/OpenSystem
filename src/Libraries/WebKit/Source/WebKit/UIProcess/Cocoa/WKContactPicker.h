/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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
#pragma once

#if HAVE(CONTACTSUI)

#include <optional>
#include <wtf/Forward.h>

@class WKWebView;
@protocol WKContactPickerDelegate;

namespace WebCore {
struct ContactInfo;
struct ContactsRequestData;
}

namespace WebKit {
enum class PickerDismissalReason : uint8_t;
}

@interface WKContactPicker : NSObject

- (instancetype)initWithView:(WKWebView *)view;

- (void)presentWithRequestData:(const WebCore::ContactsRequestData&)requestData completionHandler:(WTF::CompletionHandler<void(std::optional<Vector<WebCore::ContactInfo>>&&)>&&)completionHandler;

- (BOOL)dismissIfNeededWithReason:(WebKit::PickerDismissalReason)reason;

@property (nonatomic, weak) id<WKContactPickerDelegate> delegate;

@end

@protocol WKContactPickerDelegate <NSObject>
@optional
- (void)contactPickerDidPresent:(WKContactPicker *)contactPicker;
- (void)contactPickerDidDismiss:(WKContactPicker *)contactPicker;
@end

@interface WKContactPicker (WKTesting)
- (void)dismissWithContacts:(NSArray *)contacts;
@end

#endif // HAVE(CONTACTSUI)
