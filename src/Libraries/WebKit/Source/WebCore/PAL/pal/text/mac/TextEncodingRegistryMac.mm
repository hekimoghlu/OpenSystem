/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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
#import "TextEncodingRegistry.h"

#if PLATFORM(MAC)

#import <CoreServices/CoreServices.h>
#import <wtf/spi/cf/CFStringSPI.h>

namespace PAL {

CFStringEncoding webDefaultCFStringEncoding()
{
    UInt32 script = 0;
    UInt32 region = 0;
    ::TextEncoding encoding;
    OSErr err;
    ItemCount dontcare;

    // FIXME: Switch away from using Script Manager, as it does not support some languages newly added in OS X.
    // <rdar://problem/4433165> Need API that can get preferred web (and mail) encoding(s) w/o region code.
    // Alternatively, we could have our own table of preferred encodings in WebKit.
    //
    // Also, language changes do not apply to _CFStringGetUserDefaultEncoding() until re-login, which could be very confusing.

    _CFStringGetUserDefaultEncoding(&script, &region);
    err = TECGetWebTextEncodings(region, &encoding, 1, &dontcare);
    if (err != noErr)
        encoding = kCFStringEncodingISOLatin1;
    return encoding;
}

} // namespace WebCore

#endif // PLATFORM(MAC)
