/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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

#include <wtf/ListHashSet.h>
#include <wtf/text/StringView.h>

namespace WebCore {

class Element;
const int UninitializedDescriptor = -1;
const float DefaultDensityValue = 1.0;

class DescriptorParsingResult {
public:
    DescriptorParsingResult()
        : m_density(UninitializedDescriptor)
        , m_resourceWidth(UninitializedDescriptor)
        , m_resourceHeight(UninitializedDescriptor)
    {
    }

    bool hasDensity() const { return m_density >= 0; }
    bool hasWidth() const { return m_resourceWidth >= 0; }
    bool hasHeight() const { return m_resourceHeight >= 0; }

    float density() const { ASSERT(hasDensity()); return m_density; }
    unsigned resourceWidth() const { ASSERT(hasWidth()); return m_resourceWidth; }
    unsigned resourceHeight() const { ASSERT(hasHeight()); return m_resourceHeight; }

    void setResourceWidth(int width) { ASSERT(width >= 0); m_resourceWidth = (unsigned)width; }
    void setResourceHeight(int height) { ASSERT(height >= 0); m_resourceHeight = (unsigned)height; }
    void setDensity(float densityToSet) { ASSERT(densityToSet >= 0); m_density = densityToSet; }

private:
    float m_density;
    int m_resourceWidth;
    int m_resourceHeight;
};

struct ImageCandidate {
    enum OriginAttribute {
        SrcsetOrigin,
        SrcOrigin
    };

    ImageCandidate()
        : density(DefaultDensityValue)
        , resourceWidth(UninitializedDescriptor)
        , originAttribute(SrcsetOrigin)
    {
    }

    ImageCandidate(StringViewWithUnderlyingString source, const DescriptorParsingResult& result, OriginAttribute originAttribute)
        : string(source)
        , density(result.hasDensity() ? result.density() : UninitializedDescriptor)
        , resourceWidth(result.hasWidth() ? result.resourceWidth() : UninitializedDescriptor)
        , originAttribute(originAttribute)
    {
    }

    bool srcOrigin() const
    {
        return (originAttribute == SrcOrigin);
    }
    
    bool isEmpty() const
    {
        return string.view.isEmpty();
    }

    StringViewWithUnderlyingString string;
    float density;
    int resourceWidth;
    OriginAttribute originAttribute;
};

ImageCandidate bestFitSourceForImageAttributes(float deviceScaleFactor, const AtomString& srcAttribute, StringView srcsetAttribute, float sourceSize, Function<bool(const ImageCandidate&)>&& shouldIgnoreCandidateCallback = { });

Vector<ImageCandidate> parseImageCandidatesFromSrcsetAttribute(StringView attribute);
void getURLsFromSrcsetAttribute(const Element&, StringView attribute, ListHashSet<URL>&);
String replaceURLsInSrcsetAttribute(const Element&, StringView attribute, const UncheckedKeyHashMap<String, String>& replacementURLStrings);

} // namespace WebCore
