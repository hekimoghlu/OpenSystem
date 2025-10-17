/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
#import "_WKSystemPreferences.h"
#import <wtf/text/ASCIILiteral.h>

const auto LDMEnabledKey = CFSTR("LDMGlobalEnabled");

#define WK_LOCKDOWN_MODE_ENABLED_KEY_MACRO WKLockdownModeEnabled
// "WKLockdownModeEnabled"_s
constexpr auto WKLockdownModeEnabledKey = WTF_CONCAT(STRINGIZE_VALUE_OF(WK_LOCKDOWN_MODE_ENABLED_KEY_MACRO), _s);
// CFSTR("WKLockdownModeEnabled")
const auto WKLockdownModeEnabledKeyCFString = CFSTR(STRINGIZE_VALUE_OF(WK_LOCKDOWN_MODE_ENABLED_KEY_MACRO));

// This string must remain consistent with the lockdown mode notification name in privacy settings.
constexpr auto WKLockdownModeContainerConfigurationChangedNotification = @"WKCaptivePortalModeContainerConfigurationChanged";
