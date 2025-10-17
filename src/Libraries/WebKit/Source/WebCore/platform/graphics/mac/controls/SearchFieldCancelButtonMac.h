/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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

#if PLATFORM(MAC)

#import "SearchControlMac.h"

namespace WebCore {

class SearchFieldCancelButtonPart;

class SearchFieldCancelButtonMac final : public SearchControlMac {
public:
    SearchFieldCancelButtonMac(SearchFieldCancelButtonPart& owningPart, ControlFactoryMac&, NSSearchFieldCell *);

private:
    IntSize cellSize(NSControlSize, const ControlStyle&) const override;

    FloatRect rectForBounds(const FloatRect& bounds, const ControlStyle&) const override;

    void updateCellStates(const FloatRect&, const ControlStyle&) override;

    void draw(GraphicsContext&, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle&) override;
};

} // namespace WebCore

#endif // PLATFORM(MAC)
