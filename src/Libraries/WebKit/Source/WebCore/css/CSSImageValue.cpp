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
#include "config.h"
#include "CSSImageValue.h"

#include "CSSMarkup.h"
#include "CSSPrimitiveValue.h"
#include "CSSValueKeywords.h"
#include "CachedImage.h"
#include "CachedResourceLoader.h"
#include "CachedResourceRequest.h"
#include "CachedResourceRequestInitiatorTypes.h"
#include "DeprecatedCSSOMPrimitiveValue.h"
#include "Document.h"
#include "Element.h"
#include "StyleBuilderState.h"
#include "StyleCachedImage.h"

namespace WebCore {

CSSImageValue::CSSImageValue()
    : CSSValue(ClassType::Image)
    , m_isInvalid(true)
{
}

CSSImageValue::CSSImageValue(ResolvedURL&& location, LoadedFromOpaqueSource loadedFromOpaqueSource, AtomString&& initiatorType)
    : CSSValue(ClassType::Image)
    , m_location(WTFMove(location))
    , m_initiatorType(WTFMove(initiatorType))
    , m_loadedFromOpaqueSource(loadedFromOpaqueSource)
{
}

Ref<CSSImageValue> CSSImageValue::create()
{
    return adoptRef(*new CSSImageValue);
}

Ref<CSSImageValue> CSSImageValue::create(ResolvedURL location, LoadedFromOpaqueSource loadedFromOpaqueSource, AtomString initiatorType)
{
    return adoptRef(*new CSSImageValue(WTFMove(location), loadedFromOpaqueSource, WTFMove(initiatorType)));
}

Ref<CSSImageValue> CSSImageValue::create(URL imageURL, LoadedFromOpaqueSource loadedFromOpaqueSource, AtomString initiatorType)
{
    return create(makeResolvedURL(WTFMove(imageURL)), loadedFromOpaqueSource, WTFMove(initiatorType));
}

CSSImageValue::~CSSImageValue() = default;

bool CSSImageValue::isPending() const
{
    return !m_cachedImage;
}

URL CSSImageValue::reresolvedURL(const Document& document) const
{
    if (isCSSLocalURL(m_location.resolvedURL.string()))
        return m_location.resolvedURL;

    // Re-resolving the URL is important for cases where resolvedURL is still not an absolute URL.
    // This can happen if there was no absolute base URL when the value was created, like a style from a document without a base URL.
    if (m_location.isLocalURL())
        return document.completeURL(m_location.specifiedURLString, URL());

    return document.completeURL(m_location.resolvedURL.string());
}

RefPtr<StyleImage> CSSImageValue::createStyleImage(const Style::BuilderState& state) const
{
    auto location = makeResolvedURL(reresolvedURL(state.document()));
    if (m_location == location)
        return StyleCachedImage::create(const_cast<CSSImageValue&>(*this));
    auto result = create(WTFMove(location), m_loadedFromOpaqueSource);
    result->m_cachedImage = m_cachedImage;
    result->m_initiatorType = m_initiatorType;
    result->m_unresolvedValue = const_cast<CSSImageValue*>(this);
    return StyleCachedImage::create(WTFMove(result));
}

CachedImage* CSSImageValue::loadImage(CachedResourceLoader& loader, const ResourceLoaderOptions& options)
{
    if (!m_cachedImage) {
        ResourceLoaderOptions loadOptions = options;
        loadOptions.loadedFromOpaqueSource = m_loadedFromOpaqueSource;
        CachedResourceRequest request(ResourceRequest(reresolvedURL(*loader.document())), loadOptions);
        if (m_initiatorType.isEmpty())
            request.setInitiatorType(cachedResourceRequestInitiatorTypes().css);
        else
            request.setInitiatorType(m_initiatorType);
        if (options.mode == FetchOptions::Mode::Cors) {
            ASSERT(loader.document());
            request.updateForAccessControl(*loader.document());
        }
        m_cachedImage = loader.requestImage(WTFMove(request)).value_or(nullptr);
        for (auto imageValue = this; (imageValue = imageValue->m_unresolvedValue.get()); )
            imageValue->m_cachedImage = m_cachedImage;
    }
    return m_cachedImage.value().get();
}

bool CSSImageValue::customTraverseSubresources(const Function<bool(const CachedResource&)>& handler) const
{
    return m_cachedImage && *m_cachedImage && handler(**m_cachedImage);
}

void CSSImageValue::customSetReplacementURLForSubresources(const UncheckedKeyHashMap<String, String>& replacementURLStrings)
{
    auto replacementURLString = replacementURLStrings.get(m_location.resolvedURL.string());
    if (!replacementURLString.isNull())
        m_replacementURLString = replacementURLString;
    m_shouldUseResolvedURLInCSSText = true;
}

bool CSSImageValue::customMayDependOnBaseURL() const
{
    return WebCore::mayDependOnBaseURL(m_location);
}

void CSSImageValue::customClearReplacementURLForSubresources()
{
    m_replacementURLString = { };
    m_shouldUseResolvedURLInCSSText = false;
}

bool CSSImageValue::equals(const CSSImageValue& other) const
{
    return m_location == other.m_location;
}

String CSSImageValue::customCSSText() const
{
    if (m_isInvalid)
        return ""_s;

    if (!m_replacementURLString.isEmpty())
        return serializeURL(m_replacementURLString);

    if (m_shouldUseResolvedURLInCSSText)
        return serializeURL(m_location.resolvedURL.string());

    return serializeURL(m_location.specifiedURLString);
}

Ref<DeprecatedCSSOMValue> CSSImageValue::createDeprecatedCSSOMWrapper(CSSStyleDeclaration& styleDeclaration) const
{
    // We expose CSSImageValues as URI primitive values in CSSOM to maintain old behavior.
    return DeprecatedCSSOMPrimitiveValue::create(CSSPrimitiveValue::createURI(m_location.resolvedURL.string()), styleDeclaration);
}

bool CSSImageValue::knownToBeOpaque(const RenderElement& renderer) const
{
    return m_cachedImage.value_or(nullptr) && (**m_cachedImage).currentFrameKnownToBeOpaque(&renderer);
}

} // namespace WebCore
