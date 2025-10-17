/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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
#import <WebKit/WKFoundation.h>

#ifdef __cplusplus
namespace WebKit {
struct ResourceLoadInfo;
}
#endif

NS_ASSUME_NONNULL_BEGIN

// https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/webRequest/ResourceType
typedef NS_ENUM(NSInteger, _WKWebExtensionWebRequestResourceType) {
    _WKWebExtensionWebRequestResourceTypeMainFrame,
    _WKWebExtensionWebRequestResourceTypeSubframe,
    _WKWebExtensionWebRequestResourceTypeStylesheet,
    _WKWebExtensionWebRequestResourceTypeScript,
    _WKWebExtensionWebRequestResourceTypeImage,
    _WKWebExtensionWebRequestResourceTypeFont,
    _WKWebExtensionWebRequestResourceTypeObject,
    _WKWebExtensionWebRequestResourceTypeXMLHTTPRequest,
    _WKWebExtensionWebRequestResourceTypePing,
    _WKWebExtensionWebRequestResourceTypeCSPReport,
    _WKWebExtensionWebRequestResourceTypeMedia,
    _WKWebExtensionWebRequestResourceTypeWebsocket,
    _WKWebExtensionWebRequestResourceTypeApplicationManifest,
    _WKWebExtensionWebRequestResourceTypeXSLT,
    _WKWebExtensionWebRequestResourceTypeBeacon,
    _WKWebExtensionWebRequestResourceTypeOther,
};

#ifdef __cplusplus
WK_EXTERN _WKWebExtensionWebRequestResourceType toWebExtensionWebRequestResourceType(const WebKit::ResourceLoadInfo&);
#endif

// https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/webRequest/RequestFilter
WK_EXTERN
@interface _WKWebExtensionWebRequestFilter : NSObject

- (nullable instancetype)initWithDictionary:(NSDictionary<NSString *, id> *)dictionary outErrorMessage:(NSString * _Nullable * _Nonnull)outErrorMessage;

- (BOOL)matchesRequestForResourceOfType:(_WKWebExtensionWebRequestResourceType)resourceType URL:(NSURL *)URL tabID:(double)tabID windowID:(double)windowID;

@end

NS_ASSUME_NONNULL_END
