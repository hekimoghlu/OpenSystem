/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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
#include "config.h"
#include "WebCaptionPreferencesDelegate.h"

#if HAVE(MEDIA_ACCESSIBILITY_FRAMEWORK)

#include <WebCore/CaptionUserPreferencesMediaAF.h>
#include "WebProcess.h"
#include "WebProcessProxyMessages.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebCaptionPreferencesDelegate);

void WebCaptionPreferencesDelegate::setDisplayMode(WebCore::CaptionUserPreferences::CaptionDisplayMode displayMode)
{
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebProcessProxy::SetCaptionDisplayMode(displayMode), 0);
    WebCore::CaptionUserPreferencesMediaAF::setCachedCaptionDisplayMode(displayMode);
}

void WebCaptionPreferencesDelegate::setPreferredLanguage(const String& language)
{
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebProcessProxy::SetCaptionLanguage(language), 0);
}

} // namespace WebKit

#endif
