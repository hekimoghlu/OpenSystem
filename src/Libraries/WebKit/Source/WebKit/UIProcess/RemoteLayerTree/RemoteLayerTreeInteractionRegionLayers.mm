/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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
#import "RemoteLayerTreeInteractionRegionLayers.h"

#if ENABLE(GAZE_GLOW_FOR_INTERACTION_REGIONS)

#import "PlatformCALayerRemote.h"
#import "RealitySystemSupportSPI.h"
#import "RemoteLayerTreeHost.h"
#import <WebCore/WebActionDisablingCALayerDelegate.h>
#import <wtf/SoftLinking.h>
#import <wtf/text/MakeString.h>

SOFT_LINK_PRIVATE_FRAMEWORK_OPTIONAL(RealitySystemSupport)
SOFT_LINK_CLASS_OPTIONAL(RealitySystemSupport, RCPGlowEffectLayer)
SOFT_LINK_CONSTANT_MAY_FAIL(RealitySystemSupport, RCPAllowedInputTypesUserInfoKey, const NSString *)

namespace WebKit {
using namespace WebCore;

NSString *interactionRegionTypeKey = @"WKInteractionRegionType";
NSString *interactionRegionGroupNameKey = @"WKInteractionRegionGroupName";

RCPRemoteEffectInputTypes interactionRegionInputTypes = RCPRemoteEffectInputTypesAll ^ RCPRemoteEffectInputTypePointer;

static Class interactionRegionLayerClass()
{
    if (getRCPGlowEffectLayerClass())
        return getRCPGlowEffectLayerClass();
    return [CALayer class];
}

static NSDictionary *interactionRegionEffectUserInfo()
{
    static NeverDestroyed<RetainPtr<NSDictionary>> interactionRegionEffectUserInfo;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        if (canLoadRCPAllowedInputTypesUserInfoKey())
            interactionRegionEffectUserInfo.get() = @{ getRCPAllowedInputTypesUserInfoKey(): @(interactionRegionInputTypes) };
    });
    return interactionRegionEffectUserInfo.get().get();
}

static float brightnessMultiplier()
{
    static float multiplier = 1.5;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        if (auto brightnessUserDefault = [[NSUserDefaults standardUserDefaults] floatForKey:@"WKInteractionRegionBrightnessMultiplier"])
            multiplier = brightnessUserDefault;
    });
    return multiplier;
}

static void configureLayerForInteractionRegion(CALayer *layer, NSString *groupName)
{
    if (![layer isKindOfClass:getRCPGlowEffectLayerClass()])
        return;

    [(RCPGlowEffectLayer *)layer setBrightnessMultiplier:brightnessMultiplier() forInputTypes:interactionRegionInputTypes];
    [(RCPGlowEffectLayer *)layer setEffectGroupConfigurator:^void(CARemoteEffectGroup *group)
    {
        group.groupName = groupName;
        group.matched = YES;
        group.userInfo = interactionRegionEffectUserInfo();
    }];
}

static void reconfigureLayerContentHint(CALayer *layer, WebCore::InteractionRegion::ContentHint contentHint)
{
    if (![layer isKindOfClass:getRCPGlowEffectLayerClass()])
        return;

    if (contentHint == WebCore::InteractionRegion::ContentHint::Photo)
        [(RCPGlowEffectLayer *)layer setContentRenderingHints:RCPGlowEffectContentRenderingHintPhoto];
    else
        [(RCPGlowEffectLayer *)layer setContentRenderingHints:0];
}

static void configureLayerAsGuard(CALayer *layer, NSString *groupName)
{
    CARemoteEffectGroup *group = [CARemoteEffectGroup groupWithEffects:@[]];
    group.groupName = groupName;
    group.matched = YES;
    group.userInfo = interactionRegionEffectUserInfo();
    layer.remoteEffects = @[ group ];
}

static NSString *interactionRegionGroupNameForRegion(const WebCore::PlatformLayerIdentifier& layerID, const WebCore::InteractionRegion& interactionRegion)
{
    return makeString("WKInteractionRegion-"_s, interactionRegion.elementIdentifier.toUInt64());
}

static void configureRemoteEffect(CALayer *layer, WebCore::InteractionRegion::Type type, NSString *groupName)
{
    switch (type) {
    case InteractionRegion::Type::Interaction:
        configureLayerForInteractionRegion(layer, groupName);
        break;
    case InteractionRegion::Type::Guard:
        configureLayerAsGuard(layer, groupName);
        break;
    case InteractionRegion::Type::Occlusion:
        break;
    }
}

static void applyBackgroundColorForDebuggingToLayer(CALayer *layer, const WebCore::InteractionRegion& region)
{
    switch (region.type) {
    case InteractionRegion::Type::Interaction:
        if (region.contentHint == WebCore::InteractionRegion::ContentHint::Photo)
            [layer setBackgroundColor:cachedCGColor({ WebCore::SRGBA<float>(0.5, 0, 0.5, .2) }).get()];
        else
            [layer setBackgroundColor:cachedCGColor({ WebCore::SRGBA<float>(0, 1, 0, .2) }).get()];
        [layer setName:@"Interaction"];
        break;
    case InteractionRegion::Type::Guard:
        [layer setBorderColor:cachedCGColor({ WebCore::SRGBA<float>(0, 0, 1, .2) }).get()];
        [layer setBorderWidth:6];
        [layer setName:@"Guard"];
        break;
    case InteractionRegion::Type::Occlusion:
        [layer setBorderColor:cachedCGColor({ WebCore::SRGBA<float>(1, 0, 0, .2) }).get()];
        [layer setBorderWidth:6];
        [layer setName:@"Occlusion"];
        break;
    }
}

static CALayer *createInteractionRegionLayer(WebCore::InteractionRegion::Type type, NSString *groupName)
{
    CALayer *layer = type == InteractionRegion::Type::Interaction
        ? [[interactionRegionLayerClass() alloc] init]
        : [[CALayer alloc] init];

    [layer setHitTestsAsOpaque:YES];
    [layer setDelegate:[WebActionDisablingCALayerDelegate shared]];

    [layer setValue:@(static_cast<uint8_t>(type)) forKey:interactionRegionTypeKey];
    [layer setValue:groupName forKey:interactionRegionGroupNameKey];

    configureRemoteEffect(layer, type, groupName);

    return layer;
}

static std::optional<WebCore::InteractionRegion::Type> interactionRegionTypeForLayer(CALayer *layer)
{
    id value = [layer valueForKey:interactionRegionTypeKey];
    if (value)
        return static_cast<InteractionRegion::Type>([value intValue]);
    return std::nullopt;
}

static NSString *interactionRegionGroupNameForLayer(CALayer *layer)
{
    return [layer valueForKey:interactionRegionGroupNameKey];
}

static CACornerMask convertToCACornerMask(OptionSet<InteractionRegion::CornerMask> mask)
{
    CACornerMask cornerMask = 0;

    if (mask.contains(InteractionRegion::CornerMask::MinXMinYCorner))
        cornerMask |= kCALayerMinXMinYCorner;
    if (mask.contains(InteractionRegion::CornerMask::MaxXMinYCorner))
        cornerMask |= kCALayerMaxXMinYCorner;
    if (mask.contains(InteractionRegion::CornerMask::MinXMaxYCorner))
        cornerMask |= kCALayerMinXMaxYCorner;
    if (mask.contains(InteractionRegion::CornerMask::MaxXMaxYCorner))
        cornerMask |= kCALayerMaxXMaxYCorner;

    return cornerMask;
}

void updateLayersForInteractionRegions(RemoteLayerTreeNode& node)
{
    ASSERT(node.uiView());

    if (node.eventRegion().interactionRegions().isEmpty() || !node.uiView()) {
        node.removeInteractionRegionsContainer();
        return;
    }

    CALayer *container = node.ensureInteractionRegionsContainer();

    HashMap<std::pair<IntRect, InteractionRegion::Type>, CALayer *>existingLayers;
    HashMap<std::pair<String, InteractionRegion::Type>, CALayer *>reusableLayers;
    for (CALayer *sublayer in container.sublayers) {
        if (auto type = interactionRegionTypeForLayer(sublayer)) {
            auto result = existingLayers.add(std::make_pair(enclosingIntRect(sublayer.frame), *type), sublayer);
            ASSERT_UNUSED(result, result.isNewEntry);

            auto reuseKey = std::make_pair(interactionRegionGroupNameForLayer(sublayer), *type);
            if (reusableLayers.contains(reuseKey))
                reusableLayers.remove(reuseKey);
            else {
                auto result = reusableLayers.add(reuseKey, sublayer);
                ASSERT_UNUSED(result, result.isNewEntry);
            }
        }
    }

    bool applyBackgroundColorForDebugging = [[NSUserDefaults standardUserDefaults] boolForKey:@"WKInteractionRegionDebugFill"];

    NSUInteger insertionPoint = 0;
    HashSet<std::pair<IntRect, InteractionRegion::Type>>dedupeSet;
    for (const WebCore::InteractionRegion& region : node.eventRegion().interactionRegions()) {
        auto rect = region.rectInLayerCoordinates;
        if (!node.visibleRect() || !node.visibleRect()->intersects(rect))
            continue;

        auto interactionRegionGroupName = interactionRegionGroupNameForRegion(node.layerID(), region);
        auto key = std::make_pair(enclosingIntRect(rect), region.type);
        if (dedupeSet.contains(key))
            continue;
        auto reuseKey = std::make_pair(interactionRegionGroupName, region.type);

        RetainPtr<CALayer> regionLayer;
        bool didReuseLayer = true;
        bool didReuseLayerBasedOnRect = false;

        auto findOrCreateLayer = [&]() {
            auto layerIterator = existingLayers.find(key);
            if (layerIterator != existingLayers.end()) {
                didReuseLayerBasedOnRect = true;
                regionLayer = layerIterator->value;
                return;
            }

            auto layerReuseIterator = reusableLayers.find(reuseKey);
            if (layerReuseIterator != reusableLayers.end()) {
                regionLayer = layerReuseIterator->value;
                return;
            }

            didReuseLayer = false;
            regionLayer = adoptNS(createInteractionRegionLayer(region.type, interactionRegionGroupName));
        };
        findOrCreateLayer();

        if (didReuseLayer) {
            auto layerKey = std::make_pair(enclosingIntRect([regionLayer frame]), region.type);
            auto reuseKey = std::make_pair(interactionRegionGroupNameForLayer(regionLayer.get()), region.type);
            existingLayers.remove(layerKey);
            reusableLayers.remove(reuseKey);

            bool shouldReconfigureRemoteEffect = didReuseLayerBasedOnRect && ![interactionRegionGroupName isEqualToString:interactionRegionGroupNameForLayer(regionLayer.get())];
            if (shouldReconfigureRemoteEffect)
                configureRemoteEffect(regionLayer.get(), region.type, interactionRegionGroupName);
        }

        if (!didReuseLayerBasedOnRect)
            [regionLayer setFrame:rect];

        if (region.type == InteractionRegion::Type::Interaction) {
            [regionLayer setCornerRadius:region.cornerRadius];
            if (region.cornerRadius)
                [regionLayer setCornerCurve:kCACornerCurveCircular];
            reconfigureLayerContentHint(regionLayer.get(), region.contentHint);
            constexpr CACornerMask allCorners = kCALayerMinXMinYCorner | kCALayerMaxXMinYCorner | kCALayerMinXMaxYCorner | kCALayerMaxXMaxYCorner;
            if (region.maskedCorners.isEmpty())
                [regionLayer setMaskedCorners:allCorners];
            else
                [regionLayer setMaskedCorners:convertToCACornerMask(region.maskedCorners)];

            if (region.clipPath) {
                RetainPtr<CAShapeLayer> mask = [regionLayer mask];
                if (!mask) {
                    mask = adoptNS([[CAShapeLayer alloc] init]);
                    [regionLayer setMask:mask.get()];
                }

                [mask setFrame:[regionLayer bounds]];
                [mask setPath:region.clipPath->platformPath()];
            } else
                [regionLayer setMask:nil];
        }

        if (applyBackgroundColorForDebugging)
            applyBackgroundColorForDebuggingToLayer(regionLayer.get(), region);

        // Since we insert new layers as we go, insertionPoint is always <= container.sublayers.count.
        ASSERT(insertionPoint <= container.sublayers.count);
        bool shouldAppendLayer = insertionPoint == container.sublayers.count;
        if (shouldAppendLayer || [container.sublayers objectAtIndex:insertionPoint] != regionLayer) {
            [regionLayer removeFromSuperlayer];
            [container insertSublayer:regionLayer.get() atIndex:insertionPoint];
        }

        insertionPoint++;
    }

    for (CALayer *sublayer : existingLayers.values())
        [sublayer removeFromSuperlayer];
}

} // namespace WebKit

#endif // ENABLE(GAZE_GLOW_FOR_INTERACTION_REGIONS)
