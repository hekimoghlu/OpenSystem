/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 24, 2024.
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

#if PLATFORM(COCOA)

#import <Foundation/Foundation.h>
#import <wtf/text/WTFString.h>

OBJC_CLASS NSLocale;

namespace WebKit {

class CoreIPCLocale {
public:
    static bool isValidIdentifier(const String&);

    CoreIPCLocale(NSLocale *);
    CoreIPCLocale(String&&);

    RetainPtr<id> toID() const;

    String identfier() const
    {
        return m_identifier;
    }

private:
    static std::optional<String> canonicalLocaleStringReplacement(const String& identifier);

    String m_identifier;
};

}

#endif // PLATFORM(COCOA)
