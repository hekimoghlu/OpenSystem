/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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

#if (PLATFORM(IOS_FAMILY) || (PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE))) && ENABLE(VIDEO)

#include "TextTrackRepresentation.h"
#include <QuartzCore/CALayer.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class TextTrackRepresentationCocoa;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::TextTrackRepresentationCocoa> : std::true_type { };
}

@class WebCoreTextTrackRepresentationCocoaHelper;

namespace WebCore {

class HTMLMediaElement;

class TextTrackRepresentationCocoa : public TextTrackRepresentation, public CanMakeWeakPtr<TextTrackRepresentationCocoa, WeakPtrFactoryInitialization::Eager> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(TextTrackRepresentationCocoa, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT explicit TextTrackRepresentationCocoa(TextTrackRepresentationClient&);
    WEBCORE_EXPORT virtual ~TextTrackRepresentationCocoa();

    TextTrackRepresentationClient& client() const { return m_client; }

    PlatformLayer* platformLayer() final { return m_layer.get(); }

    WEBCORE_EXPORT void setBounds(const IntRect&) override;
    WEBCORE_EXPORT IntRect bounds() const override;
    void boundsChanged();

    using TextTrackRepresentationFactory = WTF::Function<std::unique_ptr<TextTrackRepresentation>(TextTrackRepresentationClient&, HTMLMediaElement&)>;

    WEBCORE_EXPORT static TextTrackRepresentationFactory& representationFactory();

protected:
    // TextTrackRepresentation
    WEBCORE_EXPORT void update() override;
    WEBCORE_EXPORT void setContentScale(float) override;
    WEBCORE_EXPORT void setHidden(bool) const override;

    TextTrackRepresentationClient& m_client;

private:
    RetainPtr<CALayer> m_layer;
    RetainPtr<WebCoreTextTrackRepresentationCocoaHelper> m_delegate;
};

}

#endif // (PLATFORM(IOS_FAMILY) || (PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE))) && ENABLE(VIDEO)
