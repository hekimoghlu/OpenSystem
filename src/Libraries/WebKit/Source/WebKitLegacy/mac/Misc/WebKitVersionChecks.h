/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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
#import <wtf/spi/darwin/dyldSPI.h>

/*
    Version numbers are based on the 'current library version' specified in the WebKit build rules.
    All of these methods return or take version numbers with each part shifted to the left 2 bytes.
    For example the version 1.2.3 is returned as 0x00010203 and version 200.3.5 is returned as 0x00C80305
    A version of -1 is returned if the main executable did not link against WebKit.

    Please use the current WebKit version number, available in Configurations/Version.xcconfig,
    when adding a new version constant.
*/

#if !PLATFORM(IOS_FAMILY)
#define WEBKIT_FIRST_VERSION_WITH_3_0_CONTEXT_MENU_TAGS 0x020A0000 // 522.0.0
#define WEBKIT_FIRST_VERSION_WITH_LOCAL_RESOURCE_SECURITY_RESTRICTION 0x020A0000 // 522.0.0
#define WEBKIT_FIRST_VERSION_WITHOUT_QUICKBOOKS_QUIRK 0x020A0000 // 522.0.0
#define WEBKIT_FIRST_VERSION_WITH_MAIN_THREAD_EXCEPTIONS 0x020A0000 // 522.0.0
#define WEBKIT_FIRST_VERSION_WITHOUT_ADOBE_INSTALLER_QUIRK 0x020A0000 // 522.0.0
#define WEBKIT_FIRST_VERSION_WITH_INSPECT_ELEMENT_MENU_TAG 0x020A0C00 // 522.12.0
#define WEBKIT_FIRST_VERSION_WITH_CACHE_MODEL_API 0x020B0500 // 523.5.0
#define WEBKIT_FIRST_VERSION_WITHOUT_JAVASCRIPT_RETURN_QUIRK 0x020D0100 // 525.1.0
#define WEBKIT_FIRST_VERSION_WITH_IE_COMPATIBLE_KEYBOARD_EVENT_DISPATCH 0x020D0100 // 525.1.0
#define WEBKIT_FIRST_VERSION_WITH_LOADING_DURING_COMMON_RUNLOOP_MODES 0x020E0000 // 526.0.0
#define WEBKIT_FIRST_VERSION_WITH_MORE_STRICT_LOCAL_RESOURCE_SECURITY_RESTRICTION 0x02100200 // 528.2.0
#define WEBKIT_FIRST_VERSION_WITHOUT_WEBVIEW_INIT_THREAD_WORKAROUND 0x02100700 // 528.7.0
#define WEBKIT_FIRST_VERSION_WITH_ROUND_TWO_MAIN_THREAD_EXCEPTIONS 0x02120400 // 530.4.0
#define WEBKIT_FIRST_VERSION_WITH_ROUND_THREE_MAIN_THREAD_EXCEPTIONS 0x025A0113 // 602.1.19
#define WEBKIT_FIRST_VERSION_WITHOUT_BUMPERCAR_BACK_FORWARD_QUIRK 0x02120700 // 530.7.0
#define WEBKIT_FIRST_VERSION_WITHOUT_CONTENT_SNIFFING_FOR_FILE_URLS 0x02120A00 // 530.10.0
#define WEBKIT_FIRST_VERSION_WITHOUT_LINK_ELEMENT_TEXT_CSS_QUIRK 0x02130200 // 531.2.0
#define WEBKIT_FIRST_VERSION_WITH_CSS_ATTRIBUTE_SETTERS_IGNORING_PRIORITY 0x02170D00 // 535.13.0
#define WEBKIT_FIRST_VERSION_WITH_INSECURE_CONTENT_BLOCKING 0x02590116 // 601.1.22
#define WEBKIT_FIRST_VERSION_WITH_DEFAULT_ICON_LOADING 0x025C0126 // 604.1.38

#else
// <rdar://problem/6627758> Need to implement WebKitLinkedOnOrAfter
// Actually UIKit version numbers, since applications don't link against WebKit
#define WEBKIT_FIRST_VERSION_WITH_LOADING_DURING_COMMON_RUNLOOP_MODES 2665 // iOS 7.0
#define WEBKIT_FIRST_VERSION_WITH_INSECURE_CONTENT_BLOCKING 3454
#define WEBKIT_FIRST_VERSION_WITH_CONTENT_SECURITY_POLICY_SOURCE_STAR_PROTOCOL_RESTRICTION 3555
#endif // PLATFORM(IOS_FAMILY)

#ifdef __cplusplus
extern "C" {
#endif

BOOL WebKitLinkedOnOrAfter(int version);
void setWebKitLinkTimeVersion(int);

#ifdef __cplusplus
}
#endif
