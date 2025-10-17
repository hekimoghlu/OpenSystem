/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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

#include "HTMLDivElement.h"
#include <wtf/Forward.h>

namespace WebCore {

class HTMLProgressElement;

class ProgressShadowElement : public HTMLDivElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ProgressShadowElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ProgressShadowElement);
public:
    HTMLProgressElement* progressElement() const;

protected:
    explicit ProgressShadowElement(Document&);

private:
    bool rendererIsNeeded(const RenderStyle&) override;
};

// The subclasses of ProgressShadowElement share the same isoheap, because they don't add any more
// fields to the class.

class ProgressInnerElement final : public ProgressShadowElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ProgressInnerElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ProgressInnerElement);
public:
    static Ref<ProgressInnerElement> create(Document&);

private:
    ProgressInnerElement(Document&);

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;
    bool rendererIsNeeded(const RenderStyle&) override;
};
static_assert(sizeof(ProgressInnerElement) == sizeof(ProgressShadowElement));

class ProgressBarElement final : public ProgressShadowElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ProgressBarElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ProgressBarElement);
public:
    static Ref<ProgressBarElement> create(Document&);

private:
    ProgressBarElement(Document&);
};
static_assert(sizeof(ProgressBarElement) == sizeof(ProgressShadowElement));

class ProgressValueElement final : public ProgressShadowElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ProgressValueElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ProgressValueElement);
public:
    static Ref<ProgressValueElement> create(Document&);
    void setInlineSizePercentage(double);

private:
    ProgressValueElement(Document&);
};
static_assert(sizeof(ProgressValueElement) == sizeof(ProgressShadowElement));

} // namespace WebCore
