/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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
#import "GlobalFindInPageState.h"

#import <wtf/text/WTFString.h>

#if PLATFORM(MAC)
#import <AppKit/NSPasteboard.h>
#import <WebCore/LegacyNSPasteboardTypes.h>
#endif

namespace WebKit {

#if PLATFORM(MAC)

static NSPasteboard *findPasteboard()
{
    return [NSPasteboard pasteboardWithName:NSPasteboardNameFind];
}

#else

static String& globalStringForFind()
{
    static NeverDestroyed<String> string;
    return string.get();
}

#endif

void updateStringForFind(const String& string)
{
#if PLATFORM(MAC)
    [findPasteboard() setString:string forType:WebCore::legacyStringPasteboardType()];
#else
    globalStringForFind() = string;
#endif
}

String stringForFind()
{
#if PLATFORM(MAC)
    return [findPasteboard() stringForType:WebCore::legacyStringPasteboardType()];
#else
    return globalStringForFind();
#endif
}

} // namespace WebKit
