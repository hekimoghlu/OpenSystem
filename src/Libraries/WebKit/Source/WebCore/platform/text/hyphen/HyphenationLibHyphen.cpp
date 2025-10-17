/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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
#include "config.h"
#include "Hyphenation.h"

#if USE(LIBHYPHEN)

#include <hyphen.h>
#include <limits>
#include <stdlib.h>
#include <wtf/FileSystem.h>
#include <wtf/HashMap.h>
#include <wtf/Language.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/TinyLRUCache.h>
#include <wtf/text/AtomStringHash.h>
#include <wtf/text/CString.h>
#include <wtf/text/StringView.h>

#if PLATFORM(GTK)
#include <wtf/glib/GUniquePtr.h>
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GLib port

namespace WebCore {

static const char* const gDictionaryDirectories[] = {
    "/usr/share/hyphen",
    "/usr/local/share/hyphen",
};

static AtomString extractLocaleFromDictionaryFileName(const String& fileName)
{
    if (!fileName.startsWith("hyph_"_s) || !fileName.endsWith(".dic"_s))
        return nullAtom();

    // Dictionary files always have the form "hyph_<locale name>.dic"
    // so we strip everything except the locale.
    constexpr int prefixLength = 5;
    constexpr int suffixLength = 4;
    return StringView(fileName).substring(prefixLength, fileName.length() - prefixLength - suffixLength).convertToASCIILowercaseAtom();
}

static void scanDirectoryForDictionaries(const char* directoryPath, UncheckedKeyHashMap<AtomString, Vector<String>>& availableLocales)
{
    auto directoryPathString = String::fromUTF8(directoryPath);
    for (auto& fileName : FileSystem::listDirectory(directoryPathString)) {
        auto locale = extractLocaleFromDictionaryFileName(fileName);
        if (locale.isEmpty())
            continue;

        auto fullPath = FileSystem::pathByAppendingComponent(directoryPathString, fileName);
        auto filePath = FileSystem::realPath(fullPath);

        availableLocales.add(locale, Vector<String>()).iterator->value.append(filePath);

        String localeReplacingUnderscores = makeStringByReplacingAll(locale, '_', '-');
        if (locale != localeReplacingUnderscores)
            availableLocales.add(AtomString { localeReplacingUnderscores }, Vector<String>()).iterator->value.append(filePath);

        size_t dividerPosition = localeReplacingUnderscores.find('-');
        if (dividerPosition != notFound)
            availableLocales.add(StringView(localeReplacingUnderscores).left(dividerPosition).toAtomString(), Vector<String>()).iterator->value.append(filePath);
    }
}

#if ENABLE(DEVELOPER_MODE)

#if PLATFORM(GTK)
static CString webkitBuildDirectory()
{
    const char* webkitOutputDir = g_getenv("WEBKIT_OUTPUTDIR");
    if (webkitOutputDir)
        return webkitOutputDir;

    GUniquePtr<char> outputDir(g_build_filename(FileSystem::webkitTopLevelDirectory().data(), "WebKitBuild", nullptr));
    return outputDir.get();
}
#endif // PLATFORM(GTK)

static void scanTestDictionariesDirectoryIfNecessary(UncheckedKeyHashMap<AtomString, Vector<String>>& availableLocales)
{
    // It's unfortunate that we need to look for the dictionaries this way, but
    // libhyphen doesn't have the concept of installed dictionaries. Instead,
    // we have this special case for WebKit tests.
#if PLATFORM(GTK)
    // Try alternative dictionaries path for people using Flatpak.
    GUniquePtr<char> dictionariesPath(g_build_filename("/usr", "share", "webkitgtk-test-dicts", nullptr));
    if (g_getenv("FLATPAK_ID") && g_file_test(dictionariesPath.get(), static_cast<GFileTest>(G_FILE_TEST_IS_DIR))) {
        scanDirectoryForDictionaries(dictionariesPath.get(), availableLocales);
        return;
    }

    CString buildDirectory = webkitBuildDirectory();
    dictionariesPath.reset(g_build_filename(buildDirectory.data(), "DependenciesGTK", "Root", "webkitgtk-test-dicts", nullptr));
    if (g_file_test(dictionariesPath.get(), static_cast<GFileTest>(G_FILE_TEST_IS_DIR))) {
        scanDirectoryForDictionaries(dictionariesPath.get(), availableLocales);
        return;
    }

    // Try alternative dictionaries path for people not using JHBuild.
    dictionariesPath.reset(g_build_filename(buildDirectory.data(), "webkitgtk-test-dicts", nullptr));
    if (g_file_test(dictionariesPath.get(), static_cast<GFileTest>(G_FILE_TEST_IS_DIR)))
        scanDirectoryForDictionaries(dictionariesPath.get(), availableLocales);

#elif defined(TEST_HYPHENATAION_PATH)
    scanDirectoryForDictionaries(TEST_HYPHENATAION_PATH, availableLocales);
#else
    UNUSED_PARAM(availableLocales);
#endif
}
#endif

static UncheckedKeyHashMap<AtomString, Vector<String>>& availableLocales()
{
    static bool scannedLocales = false;
    static UncheckedKeyHashMap<AtomString, Vector<String>> availableLocales;

    if (!scannedLocales) {
        for (size_t i = 0; i < std::size(gDictionaryDirectories); i++)
            scanDirectoryForDictionaries(gDictionaryDirectories[i], availableLocales);

#if ENABLE(DEVELOPER_MODE)
        scanTestDictionariesDirectoryIfNecessary(availableLocales);
#endif

        scannedLocales = true;
    }

    return availableLocales;
}

bool canHyphenate(const AtomString& localeIdentifier)
{
    if (localeIdentifier.isNull())
        return false;
    return availableLocales().contains(localeIdentifier.convertToASCIILowercase());
}

class HyphenationDictionary : public RefCounted<HyphenationDictionary> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(HyphenationDictionary);
    WTF_MAKE_NONCOPYABLE(HyphenationDictionary);
public:
    typedef std::unique_ptr<HyphenDict, void(*)(HyphenDict*)> HyphenDictUniquePtr;

    virtual ~HyphenationDictionary() = default;

    static Ref<HyphenationDictionary> createNull()
    {
        return adoptRef(*new HyphenationDictionary());
    }

    static Ref<HyphenationDictionary> create(const CString& dictPath)
    {
        return adoptRef(*new HyphenationDictionary(dictPath));
    }

    HyphenDict* libhyphenDictionary() const
    {
        return m_libhyphenDictionary.get();
    }

private:
    HyphenationDictionary(const CString& dictPath)
        : m_libhyphenDictionary(HyphenDictUniquePtr(hnj_hyphen_load(dictPath.data()), hnj_hyphen_free))
    {
    }

    HyphenationDictionary()
        : m_libhyphenDictionary(HyphenDictUniquePtr(nullptr, hnj_hyphen_free))
    {
    }

    HyphenDictUniquePtr m_libhyphenDictionary;
};

} // namespace WebCore

namespace WTF {

template<>
class TinyLRUCachePolicy<AtomString, RefPtr<WebCore::HyphenationDictionary>>
{
public:
    static TinyLRUCache<AtomString, RefPtr<WebCore::HyphenationDictionary>, 32>& cache()
    {
        static NeverDestroyed<TinyLRUCache<AtomString, RefPtr<WebCore::HyphenationDictionary>, 32>> cache;
        return cache;
    }

    static bool isKeyNull(const AtomString& localeIdentifier)
    {
        return localeIdentifier.isNull();
    }

    static RefPtr<WebCore::HyphenationDictionary> createValueForNullKey()
    {
        return WebCore::HyphenationDictionary::createNull();
    }

    static RefPtr<WebCore::HyphenationDictionary> createValueForKey(const AtomString& dictionaryPath)
    {
        return WebCore::HyphenationDictionary::create(FileSystem::fileSystemRepresentation(dictionaryPath.string()));
    }

    static AtomString createKeyForStorage(const AtomString& key) { return key; }
};

} // namespace WTF

namespace WebCore {

static void countLeadingSpaces(const CString& utf8String, int32_t& pointerOffset, int32_t& characterOffset)
{
    pointerOffset = 0;
    characterOffset = 0;
    const char* stringData = utf8String.data();
    char32_t character = 0;
    while (static_cast<unsigned>(pointerOffset) < utf8String.length()) {
        int32_t nextPointerOffset = pointerOffset;
        U8_NEXT(stringData, nextPointerOffset, static_cast<int32_t>(utf8String.length()), character);
        if (character == static_cast<char32_t>(U_SENTINEL) || !u_isUWhiteSpace(character))
            return;

        pointerOffset = nextPointerOffset;
        characterOffset++;
    }
}

size_t lastHyphenLocation(StringView string, size_t beforeIndex, const AtomString& localeIdentifier)
{
    // libhyphen accepts strings in UTF-8 format, but WebCore can only provide StringView
    // which stores either UTF-16 or Latin1 data. This is unfortunate for performance
    // reasons and we should consider switching to a more flexible hyphenation library
    // if it is available.
    CString utf8StringCopy = string.utf8();

    // WebCore often passes strings like " wordtohyphenate" to the platform layer. Since
    // libhyphen isn't advanced enough to deal with leading spaces (presumably CoreFoundation
    // can), we should find the appropriate indexes into the string to skip them.
    int32_t leadingSpaceBytes;
    int32_t leadingSpaceCharacters;
    countLeadingSpaces(utf8StringCopy, leadingSpaceBytes, leadingSpaceCharacters);

    // The libhyphen documentation specifies that this array should be 5 bytes longer than
    // the byte length of the input string.
    Vector<char> hyphenArray(utf8StringCopy.length() - leadingSpaceBytes + 5);
    char* hyphenArrayData = hyphenArray.data();

    AtomString lowercaseLocaleIdentifier = localeIdentifier.convertToASCIILowercase();

    // Web content may specify strings for locales which do not exist or that we do not have.
    if (!availableLocales().contains(lowercaseLocaleIdentifier))
        return 0;

    for (const auto& dictionaryPath : availableLocales().get(lowercaseLocaleIdentifier)) {
        RefPtr<HyphenationDictionary> dictionary = TinyLRUCachePolicy<AtomString, RefPtr<HyphenationDictionary>>::cache().get(AtomString(dictionaryPath));

        char** replacements = nullptr;
        int* positions = nullptr;
        int* removedCharacterCounts = nullptr;
        hnj_hyphen_hyphenate2(dictionary->libhyphenDictionary(),
            utf8StringCopy.data() + leadingSpaceBytes,
            utf8StringCopy.length() - leadingSpaceBytes,
            hyphenArrayData,
            nullptr, /* output parameter for hyphenated word */
            &replacements,
            &positions,
            &removedCharacterCounts);

        if (replacements) {
            for (unsigned i = 0; i < utf8StringCopy.length() - leadingSpaceBytes - 1; i++)
                free(replacements[i]);
            free(replacements);
        }

        free(positions);
        free(removedCharacterCounts);

        for (int i = beforeIndex - leadingSpaceCharacters - 2; i >= 0; i--) {
            // libhyphen will put an odd number in hyphenArrayData at all
            // hyphenation points. A number & 1 will be true for odd numbers.
            if (hyphenArrayData[i] & 1)
                return i + 1 + leadingSpaceCharacters;
        }
    }

    return 0;
}

} // namespace WebCore

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // USE(LIBHYPHEN)
