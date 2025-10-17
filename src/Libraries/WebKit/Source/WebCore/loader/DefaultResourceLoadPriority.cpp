/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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
#include "DefaultResourceLoadPriority.h"

namespace WebCore {

ResourceLoadPriority DefaultResourceLoadPriority::forResourceType(CachedResource::Type type)
{
    switch (type) {
    case CachedResource::Type::MainResource:
        return ResourceLoadPriority::VeryHigh;
    case CachedResource::Type::CSSStyleSheet:
    case CachedResource::Type::Script:
        return ResourceLoadPriority::High;
    case CachedResource::Type::SVGFontResource:
    case CachedResource::Type::MediaResource:
    case CachedResource::Type::FontResource:
    case CachedResource::Type::RawResource:
    case CachedResource::Type::Icon:
        return ResourceLoadPriority::Medium;
    case CachedResource::Type::ImageResource:
        return ResourceLoadPriority::Low;
#if ENABLE(XSLT)
    case CachedResource::Type::XSLStyleSheet:
        return ResourceLoadPriority::High;
#endif
    case CachedResource::Type::SVGDocumentResource:
        return ResourceLoadPriority::Low;
    case CachedResource::Type::Beacon:
    case CachedResource::Type::Ping:
        return ResourceLoadPriority::VeryLow;
    case CachedResource::Type::LinkPrefetch:
        return ResourceLoadPriority::VeryLow;
#if ENABLE(VIDEO)
    case CachedResource::Type::TextTrackResource:
        return ResourceLoadPriority::Low;
#endif
#if ENABLE(MODEL_ELEMENT)
    case CachedResource::Type::EnvironmentMapResource:
    case CachedResource::Type::ModelResource:
        return ResourceLoadPriority::Medium;
#endif
#if ENABLE(APPLICATION_MANIFEST)
    case CachedResource::Type::ApplicationManifest:
        return ResourceLoadPriority::Low;
#endif
    }
    ASSERT_NOT_REACHED();
    return ResourceLoadPriority::Low;
}

}
