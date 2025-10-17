/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 19, 2022.
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
#import "SettingsBase.h"

#import <wtf/NeverDestroyed.h>

#if PLATFORM(IOS_FAMILY)
#import <pal/spi/ios/UIKitSPI.h>
#import <pal/system/ios/Device.h>
#endif

#if PLATFORM(IOS_FAMILY)
#import <pal/ios/UIKitSoftLink.h>
#endif

namespace WebCore {

#if PLATFORM(MAC)

void SettingsBase::initializeDefaultFontFamilies()
{
    setStandardFontFamily("Songti TC"_s, USCRIPT_TRADITIONAL_HAN);
    setStandardFontFamily("Songti SC"_s, USCRIPT_SIMPLIFIED_HAN);
    setStandardFontFamily("Hiragino Mincho ProN"_s, USCRIPT_KATAKANA_OR_HIRAGANA);
    setStandardFontFamily("AppleMyungjo"_s, USCRIPT_HANGUL);

    setStandardFontFamily("Times"_s, USCRIPT_COMMON);
    setFixedFontFamily("Courier"_s, USCRIPT_COMMON);
    setSerifFontFamily("Times"_s, USCRIPT_COMMON);
    setSansSerifFontFamily("Helvetica"_s, USCRIPT_COMMON);
}

#else

void SettingsBase::initializeDefaultFontFamilies()
{
    setStandardFontFamily("PingFang TC"_s, USCRIPT_TRADITIONAL_HAN);
    setStandardFontFamily("PingFang SC"_s, USCRIPT_SIMPLIFIED_HAN);
    setStandardFontFamily("Hiragino Mincho ProN"_s, USCRIPT_KATAKANA_OR_HIRAGANA);
    setStandardFontFamily("Apple SD Gothic Neo"_s, USCRIPT_HANGUL);

    setStandardFontFamily("Times"_s, USCRIPT_COMMON);
    setFixedFontFamily("Courier"_s, USCRIPT_COMMON);
    setSerifFontFamily("Times"_s, USCRIPT_COMMON);
    setSansSerifFontFamily("Helvetica"_s, USCRIPT_COMMON);
}

#endif

#if ENABLE(MEDIA_SOURCE)

bool SettingsBase::platformDefaultMediaSourceEnabled()
{
#if PLATFORM(MAC)
    return true;
#else
    return false;
#endif
}

uint64_t SettingsBase::defaultMaximumSourceBufferSize()
{
#if PLATFORM(IOS_FAMILY)
    // iOS Devices have lower memory limits, enforced by jetsam rates, and a very limited
    // ability to swap. Allow SourceBuffers to store up to 105MB each, roughly a third of
    // the limit on macOS, and approximately equivalent to the limit on Firefox.
    return 110376422;
#endif
    // For other platforms, allow SourceBuffers to store up to 304MB each, enough for approximately five minutes
    // of 1080p video and stereo audio.
    return 318767104;
}

#endif

} // namespace WebCore
