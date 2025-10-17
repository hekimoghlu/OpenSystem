/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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
#import "CoreIPCLocale.h"

#import <wtf/HashMap.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/text/StringHash.h>

namespace WebKit {

bool CoreIPCLocale::isValidIdentifier(const String& identifier)
{
    if ([[NSLocale availableLocaleIdentifiers] containsObject:identifier])
        return true;
    if (canonicalLocaleStringReplacement(identifier))
        return true;
    return false;
}

CoreIPCLocale::CoreIPCLocale(NSLocale *locale)
    : m_identifier([locale localeIdentifier])
{
}

CoreIPCLocale::CoreIPCLocale(String&& identifier)
    : m_identifier([[NSLocale currentLocale] localeIdentifier])
{
    if ([[NSLocale availableLocaleIdentifiers] containsObject:identifier])
        m_identifier = identifier;
    else if (auto fixedLocale = canonicalLocaleStringReplacement(identifier))
        m_identifier = *fixedLocale;
}

RetainPtr<id> CoreIPCLocale::toID() const
{
    return adoptNS([[NSLocale alloc] initWithLocaleIdentifier:(NSString *)m_identifier]);
}

std::optional<String> CoreIPCLocale::canonicalLocaleStringReplacement(const String& identifier)
{
    static NeverDestroyed<RetainPtr<NSDictionary>> dictionary = [] {
        RetainPtr dictionary = adoptNS([NSMutableDictionary new]);
        for (NSString *input in [NSLocale availableLocaleIdentifiers]) {
            NSString *output = [[NSLocale localeWithLocaleIdentifier:input] localeIdentifier];
            if (![output isEqualToString:input])
                [dictionary setObject:input forKey:output];
        }
        return dictionary;
    }();
    if (NSString *entry = [dictionary.get() objectForKey:identifier])
        return entry;
    return std::nullopt;
}

}
