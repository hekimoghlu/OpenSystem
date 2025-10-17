/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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
#import "config.h"
#import "WebKitSwiftOverlayMacros.h"

#if PLATFORM(MAC) || PLATFORM(MACCATALYST) || PLATFORM(IOS) || PLATFORM(VISION)

#define DEFINE_MIGRATED_SYMBOL(Symbol, macOSVersion, iOSVersion, visionOSVersion) \
    const char migrated_symbol_##Symbol = 0
FOR_EACH_MIGRATED_SWIFT_OVERLAY_SYMBOL(DEFINE_MIGRATED_SYMBOL);

char _swift_FORCE_LOAD_$_swiftWebKit = 0;

#endif // PLATFORM(MAC) || PLATFORM(MACCATALYST) || PLATFORM(IOS) || PLATFORM(VISION)

#if PLATFORM(IOS) || PLATFORM(IOS_SIMULATOR)

#define DEFINE_INSTALL_NAME(major, minor) \
    extern __attribute__((visibility ("default"))) const char install_name_ ##major## _ ##minor __asm("$ld$install_name$os" #major "." #minor "$/System/Library/PrivateFrameworks/WebKit.framework/WebKit"); \
    const char install_name_ ##major## _ ##minor = 0;

DEFINE_INSTALL_NAME(4, 3);
DEFINE_INSTALL_NAME(5, 0);
DEFINE_INSTALL_NAME(5, 1);
DEFINE_INSTALL_NAME(6, 0);
DEFINE_INSTALL_NAME(6, 1);
DEFINE_INSTALL_NAME(7, 0);
DEFINE_INSTALL_NAME(7, 1);

#endif
