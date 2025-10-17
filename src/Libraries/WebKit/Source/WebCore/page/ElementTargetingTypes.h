/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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

#include "ElementIdentifier.h"
#include "FloatPoint.h"
#include "FloatRect.h"
#include "FrameIdentifier.h"
#include "RectEdges.h"
#include "RenderStyleConstants.h"
#include "ScriptExecutionContextIdentifier.h"
#include <wtf/URLHash.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

using TargetedElementSelectors = Vector<HashSet<String>>;
using TargetedElementIdentifiers = std::pair<ElementIdentifier, ScriptExecutionContextIdentifier>;

struct TargetedElementAdjustment {
    TargetedElementIdentifiers identifiers;
    TargetedElementSelectors selectors;
};

struct TargetedElementRequest {
    std::variant<FloatPoint, String, TargetedElementSelectors> data;
    bool canIncludeNearbyElements { true };
    bool shouldIgnorePointerEventsNone { true };
};

struct TargetedElementInfo {
    ElementIdentifier elementIdentifier;
    ScriptExecutionContextIdentifier documentIdentifier;
    RectEdges<bool> offsetEdges;
    String renderedText;
    String searchableText;
    String screenReaderText;
    Vector<Vector<String>> selectors;
    FloatRect boundsInRootView;
    FloatRect boundsInClientCoordinates;
    PositionType positionType { PositionType::Static };
    Vector<FrameIdentifier> childFrameIdentifiers;
    HashSet<URL> mediaAndLinkURLs;
    bool isNearbyTarget { true };
    bool isPseudoElement { false };
    bool isInShadowTree { false };
    bool isInVisibilityAdjustmentSubtree { false };
    bool hasLargeReplacedDescendant { false };
    bool hasAudibleMedia { false };
};

} // namespace WebCore
