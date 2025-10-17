/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
#include "FontCache.h"

#include <dwrite_3.h>
#include <wtf/FileSystem.h>
#include <wtf/text/win/WCharStringExtras.h>

namespace WebCore {

static String fontsPath()
{
    const wchar_t* fontsEnvironmentVariable = L"WEBKIT_TESTFONTS";
    // <https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-getenvironmentvariable>
    // The return size includes the terminating null character.
    DWORD size = GetEnvironmentVariable(fontsEnvironmentVariable, nullptr, 0);
    if (!size)
        return { };
    Vector<UChar> buffer(size);
    // The return size doesn't include the terminating null character.
    if (GetEnvironmentVariable(fontsEnvironmentVariable, wcharFrom(buffer.data()), size) != size - 1)
        return { };
    return buffer.span().first(size - 1);
}

FontCache::CreateDWriteFactoryResult FontCache::createDWriteFactory()
{
    CreateDWriteFactoryResult result;
    COMPtr<IDWriteFactory> factory;
    HRESULT hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(&result.factory));
    if (FAILED(hr))
        return result;

    COMPtr<IDWriteFactory5> factory5(Query, result.factory);
    if (!factory5)
        return result;

    COMPtr<IDWriteFontSetBuilder1> builder;
    hr = factory5->CreateFontSetBuilder(&builder);
    if (FAILED(hr))
        return result;

    COMPtr<IDWriteFontSet> systemFontSet;
    hr = factory5->GetSystemFontSet(&systemFontSet);
    if (FAILED(hr))
        return result;

    builder->AddFontSet(systemFontSet.get());

    String baseFontPath = fontsPath();
    if (baseFontPath.isEmpty())
        return result;

    const auto fontFilenames = {
        "AHEM____.TTF"_span,
        "WebKitWeightWatcher100.ttf"_span,
        "WebKitWeightWatcher200.ttf"_span,
        "WebKitWeightWatcher300.ttf"_span,
        "WebKitWeightWatcher400.ttf"_span,
        "WebKitWeightWatcher500.ttf"_span,
        "WebKitWeightWatcher600.ttf"_span,
        "WebKitWeightWatcher700.ttf"_span,
        "WebKitWeightWatcher800.ttf"_span,
        "WebKitWeightWatcher900.ttf"_span,
    };

    for (auto filename : fontFilenames) {
        String path = FileSystem::pathByAppendingComponent(baseFontPath, filename);
        COMPtr<IDWriteFontFile> file;
        hr = factory5->CreateFontFileReference(path.wideCharacters().data(), nullptr, &file);
        if (FAILED(hr))
            return result;
        builder->AddFontFile(file.get());
    }
    COMPtr<IDWriteFontSet> fontSet;
    hr = builder->CreateFontSet(&fontSet);
    if (FAILED(hr))
        return result;
    COMPtr<IDWriteFontCollection1> collection1;
    hr = factory5->CreateFontCollectionFromFontSet(fontSet.get(), &collection1);
    if (FAILED(hr))
        return result;
    result.fontCollection.query(collection1);
    return result;
}

} // namespace WebCore
