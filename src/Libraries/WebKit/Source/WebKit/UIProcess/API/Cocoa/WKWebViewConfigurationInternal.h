/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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
#import <WebKit/WKWebViewConfigurationPrivate.h>

#ifdef __cplusplus

#import "APIPageConfiguration.h"
#import "WKObject.h"
#import <wtf/Ref.h>

namespace WebKit {

template<> struct WrapperTraits<API::PageConfiguration> {
    using WrapperClass = WKWebViewConfiguration;
};

}

@interface WKWebViewConfiguration () <WKObject> {
@package
    API::ObjectStorage<API::PageConfiguration> _pageConfiguration;
}

@property (nonatomic, readonly, nullable) NSString *_applicationNameForDesktopUserAgent;

@end

#if PLATFORM(IOS_FAMILY)
_WKDragLiftDelay toDragLiftDelay(NSUInteger);
_WKDragLiftDelay toWKDragLiftDelay(WebKit::DragLiftDelay);
WebKit::DragLiftDelay fromWKDragLiftDelay(_WKDragLiftDelay);
#endif

#endif // __cplusplus

NS_ASSUME_NONNULL_BEGIN

@interface WKWebViewConfiguration ()

+ (BOOL)_isValidCustomScheme:(NSString *)urlScheme;

@end

NS_ASSUME_NONNULL_END
