/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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

#include "PlatformCAAnimationRemote.h"
#include <WebCore/PlatformCALayer.h>

namespace WebKit {

class RemoteLayerBackingStore;
class RemoteLayerBackingStoreProperties;

enum class LayerChangeIndex : size_t {
    EventRegionChanged = 40,
#if ENABLE(SCROLLING_THREAD)
    ScrollingNodeIDChanged,
#endif
#if HAVE(CORE_ANIMATION_SEPARATED_LAYERS)
    SeparatedChanged,
#if HAVE(CORE_ANIMATION_SEPARATED_PORTALS)
    SeparatedPortalChanged,
    DescendentOfSeparatedPortalChanged,
#endif
#endif
#if ENABLE(INTERACTION_REGIONS_IN_EVENT_REGION)
    VisibleRectChanged,
#endif
    ContentsFormatChanged,
#if HAVE(CORE_MATERIAL)
    AppleVisualEffectChanged,
#endif
};

enum class LayerChange : uint64_t {
    NameChanged                         = 1LLU << 0,
    TransformChanged                    = 1LLU << 1,
    SublayerTransformChanged            = 1LLU << 2,
    ShapeRoundedRectChanged             = 1LLU << 3,
    ChildrenChanged                     = 1LLU << 4,
    AnimationsChanged                   = 1LLU << 5,
    PositionChanged                     = 1LLU << 6,
    AnchorPointChanged                  = 1LLU << 7,
    BoundsChanged                       = 1LLU << 8,
    ContentsRectChanged                 = 1LLU << 9,
    BackingStoreChanged                 = 1LLU << 10,
    FiltersChanged                      = 1LLU << 11,
    ShapePathChanged                    = 1LLU << 12,
    MaskLayerChanged                    = 1LLU << 13,
    ClonedContentsChanged               = 1LLU << 14,
    TimeOffsetChanged                   = 1LLU << 15,
    SpeedChanged                        = 1LLU << 16,
    ContentsScaleChanged                = 1LLU << 17,
    CornerRadiusChanged                 = 1LLU << 18,
    BorderWidthChanged                  = 1LLU << 19,
    OpacityChanged                      = 1LLU << 20,
    BackgroundColorChanged              = 1LLU << 21,
    BorderColorChanged                  = 1LLU << 22,
    CustomAppearanceChanged             = 1LLU << 23,
    MinificationFilterChanged           = 1LLU << 24,
    MagnificationFilterChanged          = 1LLU << 25,
    BlendModeChanged                    = 1LLU << 26,
    WindRuleChanged                     = 1LLU << 27,
    VideoGravityChanged                 = 1LLU << 28,
    AntialiasesEdgesChanged             = 1LLU << 29,
    HiddenChanged                       = 1LLU << 30,
    BackingStoreAttachmentChanged       = 1LLU << 31,
    GeometryFlippedChanged              = 1LLU << 32,
    DoubleSidedChanged                  = 1LLU << 33,
    MasksToBoundsChanged                = 1LLU << 34,
    OpaqueChanged                       = 1LLU << 35,
    ContentsHiddenChanged               = 1LLU << 36,
    UserInteractionEnabledChanged       = 1LLU << 37,
    BackdropRootChanged                 = 1LLU << 38,
    BackdropRootIsOpaqueChanged         = 1LLU << 39,
    EventRegionChanged                  = 1LLU << static_cast<size_t>(LayerChangeIndex::EventRegionChanged),
#if ENABLE(SCROLLING_THREAD)
    ScrollingNodeIDChanged              = 1LLU << static_cast<size_t>(LayerChangeIndex::ScrollingNodeIDChanged),
#endif
#if HAVE(CORE_ANIMATION_SEPARATED_LAYERS)
    SeparatedChanged                    = 1LLU << static_cast<size_t>(LayerChangeIndex::SeparatedChanged),
#if HAVE(CORE_ANIMATION_SEPARATED_PORTALS)
    SeparatedPortalChanged              = 1LLU << static_cast<size_t>(LayerChangeIndex::SeparatedPortalChanged),
    DescendentOfSeparatedPortalChanged  = 1LLU << static_cast<size_t>(LayerChangeIndex::DescendentOfSeparatedPortalChanged),
#endif
#endif
#if ENABLE(INTERACTION_REGIONS_IN_EVENT_REGION)
    VisibleRectChanged                  = 1LLU << static_cast<size_t>(LayerChangeIndex::VisibleRectChanged),
#endif
    ContentsFormatChanged               = 1LLU << static_cast<size_t>(LayerChangeIndex::ContentsFormatChanged),
#if HAVE(CORE_MATERIAL)
    AppleVisualEffectChanged            = 1LLU << static_cast<size_t>(LayerChangeIndex::AppleVisualEffectChanged),
#endif
};

struct RemoteLayerBackingStoreOrProperties {
    RemoteLayerBackingStoreOrProperties() = default;
    ~RemoteLayerBackingStoreOrProperties();
    RemoteLayerBackingStoreOrProperties(RemoteLayerBackingStoreOrProperties&&) = default;
    RemoteLayerBackingStoreOrProperties& operator=(RemoteLayerBackingStoreOrProperties&&) = default;
    RemoteLayerBackingStoreOrProperties(std::unique_ptr<RemoteLayerBackingStoreProperties>&&);

    // Used in the WebContent process.
    std::unique_ptr<RemoteLayerBackingStore> store;
    // Used in the UI process.
    std::unique_ptr<RemoteLayerBackingStoreProperties> properties;
};

struct LayerProperties {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    void notePropertiesChanged(OptionSet<LayerChange> changeFlags)
    {
        changedProperties.add(changeFlags);
        everChangedProperties.add(changeFlags);
    }

    void resetChangedProperties()
    {
        changedProperties = { };
    }

    OptionSet<LayerChange> changedProperties;
    OptionSet<LayerChange> everChangedProperties;

    String name;
    std::unique_ptr<WebCore::TransformationMatrix> transform;
    std::unique_ptr<WebCore::TransformationMatrix> sublayerTransform;
    std::unique_ptr<WebCore::FloatRoundedRect> shapeRoundedRect;

    Vector<WebCore::PlatformLayerIdentifier> children;

    struct AnimationChanges {
        Vector<std::pair<String, PlatformCAAnimationRemote::Properties>> addedAnimations;
        HashSet<String> keysOfAnimationsToRemove;
#if ENABLE(THREADED_ANIMATION_RESOLUTION)
        Vector<Ref<WebCore::AcceleratedEffect>> effects;
        WebCore::AcceleratedEffectValues baseValues;
#endif
    } animationChanges;

    WebCore::FloatPoint3D position;
    WebCore::FloatPoint3D anchorPoint { 0.5, 0.5, 0 };
    WebCore::FloatRect bounds;
    WebCore::FloatRect contentsRect { 0, 0, 1, 1 };
    RemoteLayerBackingStoreOrProperties backingStoreOrProperties;
    std::unique_ptr<WebCore::FilterOperations> filters;
    WebCore::Path shapePath;
    Markable<WebCore::PlatformLayerIdentifier> maskLayerID;
    Markable<WebCore::PlatformLayerIdentifier> clonedLayerID;
    double timeOffset { 0 };
    float speed { 1 };
    float contentsScale { 1 };
    float cornerRadius { 0 };
    float borderWidth { 0 };
    float opacity { 1 };
    WebCore::Color backgroundColor { WebCore::Color::transparentBlack };
    WebCore::Color borderColor { WebCore::Color::black };
    WebCore::GraphicsLayer::CustomAppearance customAppearance { WebCore::GraphicsLayer::CustomAppearance::None };
    WebCore::PlatformCALayer::FilterType minificationFilter { WebCore::PlatformCALayer::FilterType::Linear };
    WebCore::PlatformCALayer::FilterType magnificationFilter { WebCore::PlatformCALayer::FilterType::Linear };
    WebCore::BlendMode blendMode { WebCore::BlendMode::Normal };
    WebCore::WindRule windRule { WebCore::WindRule::NonZero };
    WebCore::MediaPlayerVideoGravity videoGravity { WebCore::MediaPlayerVideoGravity::ResizeAspect };
    bool antialiasesEdges { true };
    bool hidden { false };
    bool backingStoreAttached { true };
    bool geometryFlipped { false };
    bool doubleSided { false };
    bool masksToBounds { false };
    bool opaque { false };
    bool contentsHidden { false };
    bool userInteractionEnabled { true };
    bool backdropRoot { false };
    bool backdropRootIsOpaque { false };
    WebCore::EventRegion eventRegion;

#if ENABLE(SCROLLING_THREAD)
    Markable<WebCore::ScrollingNodeID> scrollingNodeID;
#endif
#if HAVE(CORE_ANIMATION_SEPARATED_LAYERS)
    bool isSeparated { false };
#if HAVE(CORE_ANIMATION_SEPARATED_PORTALS)
    bool isSeparatedPortal { false };
    bool isDescendentOfSeparatedPortal { false };
#endif
#endif
#if ENABLE(INTERACTION_REGIONS_IN_EVENT_REGION)
    WebCore::FloatRect visibleRect;
#endif
    WebCore::ContentsFormat contentsFormat { WebCore::ContentsFormat::RGBA8 };
#if HAVE(CORE_MATERIAL)
    WebCore::AppleVisualEffect appleVisualEffect { WebCore::AppleVisualEffect::None };
#endif
};

}
