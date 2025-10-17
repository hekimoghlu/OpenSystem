/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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
#ifndef GlyphMetricsMap_h
#define GlyphMetricsMap_h

#include "Glyph.h"
#include "Path.h"
#include <array>
#include <wtf/HashMap.h>

namespace WebCore {

const float cGlyphSizeUnknown = -1;

template<class T> class GlyphMetricsMap {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(GlyphMetricsMap);
public:
    T metricsForGlyph(Glyph glyph)
    {
        return locatePage(glyph / GlyphMetricsPage::size).metricsForGlyph(glyph);
    }

    const T& existingMetricsForGlyph(Glyph glyph)
    {
        return locatePage(glyph / GlyphMetricsPage::size).existingMetricsForGlyph(glyph);
    }

    void setMetricsForGlyph(Glyph glyph, const T& metrics)
    {
        locatePage(glyph / GlyphMetricsPage::size).setMetricsForGlyph(glyph, metrics);
    }

private:
    class GlyphMetricsPage {
        WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(GlyphMetricsPage);
    public:
        static const size_t size = 16;

        GlyphMetricsPage() = default;
        explicit GlyphMetricsPage(const T& initialValue)
        {
            fill(initialValue);
        }

        void fill(const T& value)
        {
            m_metrics.fill(value);
        }

        T metricsForGlyph(Glyph glyph) const { return m_metrics[glyph % size]; }
        const T& existingMetricsForGlyph(Glyph glyph) const { return m_metrics[glyph % size]; }
        void setMetricsForGlyph(Glyph glyph, const T& metrics)
        {
            setMetricsForIndex(glyph % size, metrics);
        }

    private:
        void setMetricsForIndex(unsigned index, const T& metrics)
        {
            m_metrics[index] = metrics;
        }

        std::array<T, size> m_metrics;
    };
    
    GlyphMetricsPage& locatePage(unsigned pageNumber)
    {
        if (!pageNumber && m_filledPrimaryPage)
            return m_primaryPage;
        return locatePageSlowCase(pageNumber);
    }

    GlyphMetricsPage& locatePageSlowCase(unsigned pageNumber);
    
    static T unknownMetrics();

    bool m_filledPrimaryPage { false };
    GlyphMetricsPage m_primaryPage; // We optimize for the page that contains glyph indices 0-255.
    UncheckedKeyHashMap<int, std::unique_ptr<GlyphMetricsPage>> m_pages;
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<class T>, GlyphMetricsMap<T>);
WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<class T>, GlyphMetricsMap<T>::GlyphMetricsPage);

template<> inline float GlyphMetricsMap<float>::unknownMetrics()
{
    return cGlyphSizeUnknown;
}

template<> inline FloatRect GlyphMetricsMap<FloatRect>::unknownMetrics()
{
    return FloatRect(0, 0, cGlyphSizeUnknown, cGlyphSizeUnknown);
}

template<> inline std::optional<Path> GlyphMetricsMap<std::optional<Path>>::unknownMetrics()
{
    return std::nullopt;
}

template<class T> typename GlyphMetricsMap<T>::GlyphMetricsPage& GlyphMetricsMap<T>::locatePageSlowCase(unsigned pageNumber)
{
    if (!pageNumber) {
        ASSERT(!m_filledPrimaryPage);
        m_primaryPage.fill(unknownMetrics());
        m_filledPrimaryPage = true;
        return m_primaryPage;
    }

    return *m_pages.ensure(pageNumber, [] {
        return makeUnique<GlyphMetricsPage>(unknownMetrics());
    }).iterator->value;
}
    
} // namespace WebCore

#endif
