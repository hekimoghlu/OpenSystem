/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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

#if ENABLE(SPELLCHECK)

#include <enchant.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class TextCheckerEnchant {
    WTF_MAKE_TZONE_ALLOCATED(TextCheckerEnchant);
    WTF_MAKE_NONCOPYABLE(TextCheckerEnchant);
    friend class NeverDestroyed<TextCheckerEnchant>;
public:
    static TextCheckerEnchant& singleton();

    void ignoreWord(const String&);
    void learnWord(const String&);
    void checkSpellingOfString(const String&, int& misspellingLocation, int& misspellingLength);
    Vector<String> getGuessesForWord(const String&);
    void updateSpellCheckingLanguages(const Vector<String>& languages);
    Vector<String> loadedSpellCheckingLanguages() const;
    bool hasDictionary() const { return !m_enchantDictionaries.isEmpty(); }
    Vector<String> availableSpellCheckingLanguages() const;

private:
    TextCheckerEnchant();
    ~TextCheckerEnchant() = delete;

    void checkSpellingOfWord(const String&, int start, int end, int& misspellingLocation, int& misspellingLength);

    struct EnchantDictDeleter {
        void operator()(EnchantDict*) const;
    };

    using UniqueEnchantDict = std::unique_ptr<EnchantDict, EnchantDictDeleter>;

    EnchantBroker* m_broker;
    Vector<UniqueEnchantDict> m_enchantDictionaries;
};

} // namespace WebCore

#endif // ENABLE(SPELLCHECK)
