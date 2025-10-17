/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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

#if ENABLE(ADVANCED_PRIVACY_PROTECTIONS)

#import <pal/spi/cocoa/WebPrivacySPI.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(PAL, WebPrivacy)
SOFT_LINK_CLASS_FOR_HEADER(PAL, WPResourceRequestOptions)
SOFT_LINK_CLASS_FOR_HEADER(PAL, WPResources)
SOFT_LINK_CLASS_FOR_HEADER(PAL, WPLinkFilteringData)
SOFT_LINK_CLASS_FOR_HEADER(PAL, WPStorageAccessPromptQuirk)
SOFT_LINK_CLASS_FOR_HEADER(PAL, WPStorageAccessPromptQuirksData)
SOFT_LINK_CLASS_FOR_HEADER(PAL, WPStorageAccessUserAgentStringQuirkData)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, WebPrivacy, WPNotificationUserInfoResourceTypeKey, NSString *)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, WebPrivacy, WPResourceDataChangedNotificationName, NSNotificationName)

#endif // ENABLE(ADVANCED_PRIVACY_PROTECTIONS)
