/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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

#include "LengthPoint.h"
#include "StyleContentAlignmentData.h"
#include "StyleSelfAlignmentData.h"
#include <memory>
#include <wtf/DataRef.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class AnimationList;
class ContentData;
class FillLayer;
class ShadowData;
class StyleDeprecatedFlexibleBoxData;
class StyleFilterData;
class StyleFlexibleBoxData;
class StyleMultiColData;
class StyleTransformData;
class StyleVisitedLinkColorData;

constexpr int appearanceBitWidth = 7;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleMiscNonInheritedData);
class StyleMiscNonInheritedData : public RefCounted<StyleMiscNonInheritedData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleMiscNonInheritedData);
public:
    static Ref<StyleMiscNonInheritedData> create() { return adoptRef(*new StyleMiscNonInheritedData); }
    Ref<StyleMiscNonInheritedData> copy() const;
    ~StyleMiscNonInheritedData();

    bool operator==(const StyleMiscNonInheritedData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleMiscNonInheritedData&) const;
#endif

    bool hasOpacity() const { return opacity < 1; }
    bool hasZeroOpacity() const { return !opacity; }
    bool hasFilters() const;
    bool contentDataEquivalent(const StyleMiscNonInheritedData&) const;

    // This is here to pack in with m_refCount.
    float opacity;

    DataRef<StyleDeprecatedFlexibleBoxData> deprecatedFlexibleBox; // Flexible box properties
    DataRef<StyleFlexibleBoxData> flexibleBox;
    DataRef<StyleMultiColData> multiCol; //  CSS3 multicol properties
    DataRef<StyleFilterData> filter; // Filter operations (url, sepia, blur, etc.)
    DataRef<StyleTransformData> transform; // Transform properties (rotate, scale, skew, etc.)
    DataRef<FillLayer> mask;
    DataRef<StyleVisitedLinkColorData> visitedLinkColor;

    RefPtr<AnimationList> animations;
    RefPtr<AnimationList> transitions;
    std::unique_ptr<ContentData> content;
    std::unique_ptr<ShadowData> boxShadow; // For box-shadow decorations.
    String altText;
    double aspectRatioWidth;
    double aspectRatioHeight;
    StyleContentAlignmentData alignContent;
    StyleContentAlignmentData justifyContent;
    StyleSelfAlignmentData alignItems;
    StyleSelfAlignmentData alignSelf;
    StyleSelfAlignmentData justifyItems;
    StyleSelfAlignmentData justifySelf;
    LengthPoint objectPosition;
    int order;

    unsigned hasAttrContent : 1 { false };
    unsigned hasDisplayAffectedByAnimations : 1 { false };
#if ENABLE(DARK_MODE_CSS)
    unsigned hasExplicitlySetColorScheme : 1 { false };
#endif
    unsigned hasExplicitlySetDirection : 1 { false };
    unsigned hasExplicitlySetWritingMode : 1 { false };
    unsigned tableLayout : 1; // TableLayoutType
    unsigned aspectRatioType : 2; // AspectRatioType
    unsigned appearance : appearanceBitWidth; // StyleAppearance
    unsigned usedAppearance : appearanceBitWidth; // StyleAppearance
    unsigned textOverflow : 1; // Whether or not lines that spill out should be truncated with "..."
    unsigned userDrag : 2; // UserDrag
    unsigned objectFit : 3; // ObjectFit
    unsigned resize : 3; // Resize

private:
    StyleMiscNonInheritedData();
    StyleMiscNonInheritedData(const StyleMiscNonInheritedData&);
};

} // namespace WebCore
