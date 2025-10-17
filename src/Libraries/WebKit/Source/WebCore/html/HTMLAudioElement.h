/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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

#if ENABLE(VIDEO)

#include "HTMLMediaElement.h"

namespace WebCore {

class Document;

class HTMLAudioElement final : public HTMLMediaElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLAudioElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLAudioElement);
public:
    static Ref<HTMLAudioElement> create(const QualifiedName&, Document&, bool);
    static Ref<HTMLAudioElement> createForLegacyFactoryFunction(Document&, const AtomString& src);

private:
    HTMLAudioElement(const QualifiedName&, Document&, bool);

    PlatformMediaSession::MediaType presentationType() const final { return PlatformMediaSession::MediaType::Audio; }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::HTMLAudioElement)
    static bool isType(const WebCore::HTMLMediaElement& element) { return element.hasTagName(WebCore::HTMLNames::audioTag); }
    static bool isType(const WebCore::Element& element)
    {
        auto* mediaElement = dynamicDowncast<WebCore::HTMLMediaElement>(element);
        return mediaElement && isType(*mediaElement);
    }
    static bool isType(const WebCore::Node& node)
    {
        auto* mediaElement = dynamicDowncast<WebCore::HTMLMediaElement>(node);
        return mediaElement && isType(*mediaElement);
    }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(VIDEO)
