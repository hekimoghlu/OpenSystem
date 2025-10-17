/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 15, 2023.
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

#if ENABLE(WRITING_TOOLS)

#if USE(APPLE_INTERNAL_SDK)

#import <WritingTools/WTSession_Private.h>
#import <WritingTools/WritingTools.h>

#if PLATFORM(MAC)

using PlatformWritingToolsBehavior = NSWritingToolsBehavior;

constexpr auto PlatformWritingToolsBehaviorNone = NSWritingToolsBehaviorNone;
constexpr auto PlatformWritingToolsBehaviorDefault = NSWritingToolsBehaviorDefault;
constexpr auto PlatformWritingToolsBehaviorLimited = NSWritingToolsBehaviorLimited;
constexpr auto PlatformWritingToolsBehaviorComplete = NSWritingToolsBehaviorComplete;

// FIXME: (rdar://130540028) Remove uses of the old WritingToolsAllowedInputOptions API in favor of the new WritingToolsResultOptions API, and remove staging.

using PlatformWritingToolsResultOptions = NSUInteger;

constexpr auto PlatformWritingToolsResultPlainText = (PlatformWritingToolsResultOptions)(1 << 0);
constexpr auto PlatformWritingToolsResultRichText = (PlatformWritingToolsResultOptions)(1 << 1);
constexpr auto PlatformWritingToolsResultList = (PlatformWritingToolsResultOptions)(1 << 2);
constexpr auto PlatformWritingToolsResultTable = (PlatformWritingToolsResultOptions)(1 << 3);

#else

#import <UIKit/UIKit.h>

using PlatformWritingToolsBehavior = UIWritingToolsBehavior;

constexpr auto PlatformWritingToolsBehaviorNone = UIWritingToolsBehaviorNone;
constexpr auto PlatformWritingToolsBehaviorDefault = UIWritingToolsBehaviorDefault;
constexpr auto PlatformWritingToolsBehaviorLimited = UIWritingToolsBehaviorLimited;
constexpr auto PlatformWritingToolsBehaviorComplete = UIWritingToolsBehaviorComplete;

// FIXME: (rdar://130540028) Remove uses of the old WritingToolsAllowedInputOptions API in favor of the new WritingToolsResultOptions API, and remove staging.

using PlatformWritingToolsResultOptions = NSUInteger;

constexpr auto PlatformWritingToolsResultPlainText = (PlatformWritingToolsResultOptions)(1 << 0);
constexpr auto PlatformWritingToolsResultRichText = (PlatformWritingToolsResultOptions)(1 << 1);
constexpr auto PlatformWritingToolsResultList = (PlatformWritingToolsResultOptions)(1 << 2);
constexpr auto PlatformWritingToolsResultTable = (PlatformWritingToolsResultOptions)(1 << 3);

#endif

#else

#error Symbols must be forward declared once used with non-internal SDKS.

#endif // USE(APPLE_INTERNAL_SDK)

#endif // ENABLE(WRITING_TOOLS)
