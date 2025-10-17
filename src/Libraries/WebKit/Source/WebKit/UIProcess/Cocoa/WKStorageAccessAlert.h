/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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

#if PLATFORM(COCOA) && !PLATFORM(WATCHOS) && !PLATFORM(APPLETV)

#import <wtf/CompletionHandler.h>

@class WKWebView;

namespace WTF {
class String;
}

namespace WebCore {
class RegistrableDomain;
}

namespace WebKit {

void presentStorageAccessAlert(WKWebView *, const WebCore::RegistrableDomain& requestingDomain, const WebCore::RegistrableDomain& currentDomain, CompletionHandler<void(bool)>&&);
void presentStorageAccessAlertQuirk(WKWebView *, const WebCore::RegistrableDomain& firstRequestingDomain, const WebCore::RegistrableDomain& secondRequestingDomain, const WebCore::RegistrableDomain& current, CompletionHandler<void(bool)>&&);
void presentStorageAccessAlertSSOQuirk(WKWebView *, const String& organizationName, const HashMap<WebCore::RegistrableDomain, Vector<WebCore::RegistrableDomain>>&, CompletionHandler<void(bool)>&&);
void displayStorageAccessAlert(WKWebView *, NSString *, NSString *, NSString *, NSArray<NSString *> *, CompletionHandler<void(bool)>&&);

}

#endif // PLATFORM(COCOA) && !PLATFORM(WATCHOS) && !PLATFORM(APPLETV)
