/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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

#include "AccessibilityRenderObject.h"

namespace WebCore {

class HTMLMeterElement;
class RenderMeter;

class HTMLProgressElement;
class RenderProgress;
    
class AccessibilityProgressIndicator final : public AccessibilityRenderObject {
public:
    static Ref<AccessibilityProgressIndicator> create(AXID, RenderObject&);

    bool isIndeterminate() const final;

private:
    explicit AccessibilityProgressIndicator(AXID, RenderObject&);

    AccessibilityRole determineAccessibilityRole() final;

    String valueDescription() const final;
    String gaugeRegionValueDescription() const;
    float valueForRange() const final;
    float maxValueForRange() const final;
    float minValueForRange() const final;

    HTMLProgressElement* progressElement() const;
    HTMLMeterElement* meterElement() const;
    
    bool computeIsIgnored() const final;
};

} // namespace WebCore
