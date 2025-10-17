/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 1, 2025.
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

#include "FontPlatformData.h"
#include "RenderingResourceIdentifier.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(WIN)
#include <wtf/text/WTFString.h>
#elif USE(CORE_TEXT)
#include <CoreFoundation/CFBase.h>
#include <wtf/RetainPtr.h>

typedef struct CGFont* CGFontRef;
typedef const struct __CTFontDescriptor* CTFontDescriptorRef;
#elif USE(CAIRO)
#include "RefPtrCairo.h"

typedef struct FT_FaceRec_*  FT_Face;
#elif USE(SKIA)
#include <skia/core/SkTypeface.h>
#endif

namespace WebCore {

class SharedBuffer;
class FontDescription;
class FontCreationContext;
enum class FontTechnology : uint8_t;

template <typename T> class FontTaggedSettings;
typedef FontTaggedSettings<int> FontFeatureSettings;

struct FontCustomPlatformSerializedData {
    Ref<SharedBuffer> fontFaceData;
    String itemInCollection;
    RenderingResourceIdentifier renderingResourceIdentifier;
};

struct FontCustomPlatformData : public RefCounted<FontCustomPlatformData> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(FontCustomPlatformData);
    WTF_MAKE_NONCOPYABLE(FontCustomPlatformData);
public:
    WEBCORE_EXPORT static RefPtr<FontCustomPlatformData> create(SharedBuffer&, const String&);
    WEBCORE_EXPORT static RefPtr<FontCustomPlatformData> createMemorySafe(SharedBuffer&, const String&);

#if PLATFORM(WIN) && USE(CAIRO)
    FontCustomPlatformData(const String& name, FontPlatformData::CreationData&&);
#elif USE(CORE_TEXT)
    FontCustomPlatformData(CTFontDescriptorRef fontDescriptor, FontPlatformData::CreationData&& creationData)
        : fontDescriptor(fontDescriptor)
        , creationData(WTFMove(creationData))
        , m_renderingResourceIdentifier(RenderingResourceIdentifier::generate())
    {
    }
#elif USE(CAIRO)
    FontCustomPlatformData(FT_Face, FontPlatformData::CreationData&&);
#elif USE(SKIA)
    FontCustomPlatformData(sk_sp<SkTypeface>&&, FontPlatformData::CreationData&&);
#endif
    WEBCORE_EXPORT ~FontCustomPlatformData();

    FontPlatformData fontPlatformData(const FontDescription&, bool bold, bool italic, const FontCreationContext&);

    WEBCORE_EXPORT FontCustomPlatformSerializedData serializedData() const;
    WEBCORE_EXPORT static std::optional<Ref<FontCustomPlatformData>> tryMakeFromSerializationData(FontCustomPlatformSerializedData&&, bool);

    static bool supportsFormat(const String&);
    static bool supportsTechnology(const FontTechnology&);

#if PLATFORM(WIN) && USE(CAIRO)
    String name;
#elif USE(CORE_TEXT)
    RetainPtr<CTFontDescriptorRef> fontDescriptor;
#elif USE(CAIRO)
    RefPtr<cairo_font_face_t> m_fontFace;
#elif USE(SKIA)
    sk_sp<SkTypeface> m_typeface;
#endif
    FontPlatformData::CreationData creationData;

    RenderingResourceIdentifier m_renderingResourceIdentifier;
};

inline RefPtr<const FontCustomPlatformData> FontPlatformData::protectedCustomPlatformData() const
{
    return m_customPlatformData.get();
}

} // namespace WebCore
