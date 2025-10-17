/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 19, 2023.
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
#include "DisplayListItem.h"

#include "DisplayListItems.h"
#include "DisplayListResourceHeap.h"
#include "FilterResults.h"
#include "GraphicsContext.h"
#include <wtf/text/TextStream.h>

namespace WebCore {
namespace DisplayList {

template<typename, typename = void> inline constexpr bool HasIsValid = false;
template<typename T> inline constexpr bool HasIsValid<T, std::void_t<decltype(std::declval<T>().isValid())>> = true;

bool isValid(const Item& item)
{
    return WTF::switchOn(item, [&]<typename T> (const T& item) {
        if constexpr (HasIsValid<T>)
            return item.isValid();
        else {
            UNUSED_PARAM(item);
            return true;
        }
    });
}

template<class T>
inline static std::optional<RenderingResourceIdentifier> applyFilteredImageBufferItem(GraphicsContext& context, const ResourceHeap& resourceHeap, const T& item, OptionSet<ReplayOption> options)
{
    auto resourceIdentifier = item.sourceImageIdentifier();
    auto sourceImage = resourceIdentifier ? resourceHeap.getImageBuffer(*resourceIdentifier, options) : nullptr;
    if (UNLIKELY(!sourceImage && resourceIdentifier))
        return resourceIdentifier;

    FilterResults results;
    item.apply(context, sourceImage, results);
    return std::nullopt;
}

template<class T>
inline static std::optional<RenderingResourceIdentifier> applyImageBufferItem(GraphicsContext& context, const ResourceHeap& resourceHeap, const T& item, OptionSet<ReplayOption> options)
{
    auto resourceIdentifier = item.imageBufferIdentifier();
    if (auto* imageBuffer = resourceHeap.getImageBuffer(resourceIdentifier, options)) {
        item.apply(context, *imageBuffer);
        return std::nullopt;
    }
    return resourceIdentifier;
}

template<class T>
inline static std::optional<RenderingResourceIdentifier> applyNativeImageItem(GraphicsContext& context, const ResourceHeap& resourceHeap, const T& item, OptionSet<ReplayOption> options)
{
    auto resourceIdentifier = item.imageIdentifier();
    if (auto* image = resourceHeap.getNativeImage(resourceIdentifier, options)) {
        item.apply(context, *image);
        return std::nullopt;
    }
    return resourceIdentifier;
}

template<class T>
inline static std::optional<RenderingResourceIdentifier> applySourceImageItem(GraphicsContext& context, const ResourceHeap& resourceHeap, const T& item, OptionSet<ReplayOption> options)
{
    auto resourceIdentifier = item.imageIdentifier();
    if (auto sourceImage = resourceHeap.getSourceImage(resourceIdentifier, options)) {
        item.apply(context, *sourceImage);
        return std::nullopt;
    }
    return resourceIdentifier;
}

inline static std::optional<RenderingResourceIdentifier> applySetStateItem(GraphicsContext& context, const ResourceHeap& resourceHeap, const SetState& item, OptionSet<ReplayOption> options)
{
    auto fixPatternTileImage = [&](Pattern* pattern) -> std::optional<RenderingResourceIdentifier> {
        if (!pattern)
            return std::nullopt;

        auto imageIdentifier = pattern->tileImage().imageIdentifier();
        auto sourceImage = resourceHeap.getSourceImage(imageIdentifier, options);
        if (!sourceImage)
            return imageIdentifier;

        pattern->setTileImage(WTFMove(*sourceImage));
        return std::nullopt;
    };

    if (auto imageIdentifier = fixPatternTileImage(item.state().strokeBrush().pattern()))
        return *imageIdentifier;

    if (auto imageIdentifier = fixPatternTileImage(item.state().fillBrush().pattern()))
        return *imageIdentifier;

    item.apply(context);
    return std::nullopt;
}

inline static std::optional<RenderingResourceIdentifier> applyDrawGlyphs(GraphicsContext& context, const ResourceHeap& resourceHeap, const DrawGlyphs& item)
{
    auto resourceIdentifier = item.fontIdentifier();
    if (auto* font = resourceHeap.getFont(resourceIdentifier)) {
        item.apply(context, *font);
        return std::nullopt;
    }
    return resourceIdentifier;
}

inline static std::optional<RenderingResourceIdentifier> applyDrawDecomposedGlyphs(GraphicsContext& context, const ResourceHeap& resourceHeap, const DrawDecomposedGlyphs& item)
{
    auto fontIdentifier = item.fontIdentifier();
    auto* font = resourceHeap.getFont(fontIdentifier);
    if (!font)
        return fontIdentifier;

    auto drawGlyphsIdentifier = item.decomposedGlyphsIdentifier();
    auto* decomposedGlyphs = resourceHeap.getDecomposedGlyphs(drawGlyphsIdentifier);
    if (!decomposedGlyphs)
        return drawGlyphsIdentifier;

    item.apply(context, *font, *decomposedGlyphs);
    return std::nullopt;
}

ApplyItemResult applyItem(GraphicsContext& context, const ResourceHeap& resourceHeap, ControlFactory& controlFactory, const Item& item, OptionSet<ReplayOption> options)
{
    if (!isValid(item))
        return { StopReplayReason::InvalidItemOrExtent, std::nullopt };

    return WTF::switchOn(item,
        [&](const ClipToImageBuffer& item) -> ApplyItemResult {
            if (auto missingCachedResourceIdentifier = applyImageBufferItem(context, resourceHeap, item, options))
                return { StopReplayReason::MissingCachedResource, WTFMove(missingCachedResourceIdentifier) };
            return { };
        }, [&](const DrawControlPart& item) -> ApplyItemResult {
            item.apply(context, controlFactory);
            return { };
        }, [&](const DrawGlyphs& item) -> ApplyItemResult {
            if (auto missingCachedResourceIdentifier = applyDrawGlyphs(context, resourceHeap, item))
                return { StopReplayReason::MissingCachedResource, WTFMove(missingCachedResourceIdentifier) };
            return { };
        }, [&](const DrawDecomposedGlyphs& item) -> ApplyItemResult {
            if (auto missingCachedResourceIdentifier = applyDrawDecomposedGlyphs(context, resourceHeap, item))
                return { StopReplayReason::MissingCachedResource, WTFMove(missingCachedResourceIdentifier) };
            return { };
        }, [&](const DrawFilteredImageBuffer& item) -> ApplyItemResult {
            if (auto missingCachedResourceIdentifier = applyFilteredImageBufferItem(context, resourceHeap, item, options))
                return { StopReplayReason::MissingCachedResource, WTFMove(missingCachedResourceIdentifier) };
            return { };
        }, [&](const DrawImageBuffer& item) -> ApplyItemResult {
            if (auto missingCachedResourceIdentifier = applyImageBufferItem(context, resourceHeap, item, options))
                return { StopReplayReason::MissingCachedResource, WTFMove(missingCachedResourceIdentifier) };
            return { };
        }, [&](const DrawNativeImage& item) -> ApplyItemResult {
            if (auto missingCachedResourceIdentifier = applyNativeImageItem<DrawNativeImage>(context, resourceHeap, item, options))
                return { StopReplayReason::MissingCachedResource, WTFMove(missingCachedResourceIdentifier) };
            return { };
        }, [&](const DrawPattern& item) -> ApplyItemResult {
            if (auto missingCachedResourceIdentifier = applySourceImageItem<DrawPattern>(context, resourceHeap, item, options))
                return { StopReplayReason::MissingCachedResource, WTFMove(missingCachedResourceIdentifier) };
            return { };
        }, [&](const SetState& item) -> ApplyItemResult {
            if (auto missingCachedResourceIdentifier = applySetStateItem(context, resourceHeap, item, options))
                return { StopReplayReason::MissingCachedResource, WTFMove(missingCachedResourceIdentifier) };
            return { };
        }, [&](const auto& item) -> ApplyItemResult {
            item.apply(context);
            return { };
        }
    );
}

bool shouldDumpItem(const Item& item, OptionSet<AsTextFlag> flags)
{
    return WTF::switchOn(item,
        [&](const SetState& item) -> bool {
            if (!flags.contains(AsTextFlag::IncludePlatformOperations))
                return true;
            // FIXME: for now, only drop the item if the only state-change flags are platform-specific.
            return item.state().changes() != GraphicsContextState::Change::ShouldSubpixelQuantizeFonts;
#if USE(CG)
        }, [&](const ApplyFillPattern&) -> bool {
            return !flags.contains(AsTextFlag::IncludePlatformOperations);
        }, [&](const ApplyStrokePattern&) -> bool {
            return !flags.contains(AsTextFlag::IncludePlatformOperations);
#endif
        }, [&](const auto&) -> bool {
            return true;
        }
    );
}

void dumpItem(TextStream& ts, const Item& item, OptionSet<AsTextFlag> flags)
{
    WTF::switchOn(item, [&]<typename ItemType> (const ItemType& item) {
        ts << ItemType::name;
        item.dump(ts, flags);
    });
}

TextStream& operator<<(TextStream& ts, const Item& item)
{
    dumpItem(ts, item, { AsTextFlag::IncludePlatformOperations, AsTextFlag::IncludeResourceIdentifiers });
    return ts;
}

TextStream& operator<<(TextStream& ts, StopReplayReason reason)
{
    switch (reason) {
    case StopReplayReason::ReplayedAllItems: ts << "ReplayedAllItems"; break;
    case StopReplayReason::MissingCachedResource: ts << "MissingCachedResource"; break;
    case StopReplayReason::InvalidItemOrExtent: ts << "InvalidItemOrExtent"; break;
    case StopReplayReason::OutOfMemory: ts << "OutOfMemory"; break;
    }
    return ts;
}

} // namespace DisplayList
} // namespace WebCore
