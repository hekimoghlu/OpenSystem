/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 12, 2024.
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
#include "CSSCalcTree+ComputedStyleDependencies.h"

#include "CSSCalcTree+Traversal.h"
#include "CSSPrimitiveNumericTypes+ComputedStyleDependencies.h"
#include "CSSPropertyNames.h"
#include "CSSUnits.h"
#include "ComputedStyleDependencies.h"
#include "ContainerQueryFeatures.h"
#include "MediaQueryFeatures.h"

namespace WebCore {
namespace CSSCalc {

static void collectComputedStyleDependencies(const Child& root, ComputedStyleDependencies& dependencies)
{
    WTF::switchOn(root,
        [&](const Number&) {
            // No potential dependencies.
        },
        [&](const Percentage&) {
            // No potential dependencies.
        },
        [&](const CanonicalDimension&) {
            // No potential dependencies.
        },
        [&](const NonCanonicalDimension& root) {
            if (auto lengthUnit = CSS::toLengthUnit(root.unit))
                CSS::collectComputedStyleDependencies(dependencies, *lengthUnit);
        },
        [&](const Symbol& root) {
            if (auto lengthUnit = CSS::toLengthUnit(root.unit))
                CSS::collectComputedStyleDependencies(dependencies, *lengthUnit);
        },
        [&](const IndirectNode<MediaProgress>& root) {
            root->feature->collectComputedStyleDependencies(dependencies);
            forAllChildNodes(*root, [&](const auto& root) { collectComputedStyleDependencies(root, dependencies); });
        },
        [&](const IndirectNode<ContainerProgress>& root) {
            root->feature->collectComputedStyleDependencies(dependencies);
            forAllChildNodes(*root, [&](const auto& root) { collectComputedStyleDependencies(root, dependencies); });
        },
        [&](const IndirectNode<Anchor>& anchor) {
            dependencies.anchors = true;
            if (anchor->fallback)
                collectComputedStyleDependencies(*anchor->fallback, dependencies);
        },
        [&](const IndirectNode<AnchorSize>& anchorSize) {
            dependencies.anchors = true;
            if (anchorSize->fallback)
                collectComputedStyleDependencies(*anchorSize->fallback, dependencies);
        },
        [&](const auto& root) {
            forAllChildNodes(*root, [&](const auto& root) { collectComputedStyleDependencies(root, dependencies); });
        }
    );
}

void collectComputedStyleDependencies(const Tree& tree, ComputedStyleDependencies& dependencies)
{
    collectComputedStyleDependencies(tree.root, dependencies);
}

} // namespace CSSCalc
} // namespace WebCore
