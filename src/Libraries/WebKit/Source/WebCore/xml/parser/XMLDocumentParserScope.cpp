/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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
#include "XMLDocumentParserScope.h"

#include "CachedResourceLoader.h"
#include "XMLDocumentParser.h"

namespace WebCore {

WeakPtr<CachedResourceLoader>& XMLDocumentParserScope::currentCachedResourceLoader()
{
    static NeverDestroyed<WeakPtr<CachedResourceLoader>> currentCachedResourceLoader;
    return currentCachedResourceLoader;
}

#if ENABLE(XSLT)
XMLDocumentParserScope::XMLDocumentParserScope(CachedResourceLoader* cachedResourceLoader)
    : XMLDocumentParserScope(cachedResourceLoader, xmlGenericError, xmlStructuredError, xmlGenericErrorContext, xmlStructuredErrorContext)
{
}
#else
XMLDocumentParserScope::XMLDocumentParserScope(CachedResourceLoader* cachedResourceLoader)
    : m_oldCachedResourceLoader(currentCachedResourceLoader())
{
    initializeXMLParser();
    m_oldEntityLoader = xmlGetExternalEntityLoader();
    currentCachedResourceLoader() = cachedResourceLoader;
    xmlSetExternalEntityLoader(WebCore::externalEntityLoader);
}
#endif // ENABLE(XSLT)

#if ENABLE(XSLT)
XMLDocumentParserScope::XMLDocumentParserScope(CachedResourceLoader* cachedResourceLoader, xmlGenericErrorFunc genericErrorFunc, xmlStructuredErrorFunc structuredErrorFunc, void* genericErrorContext, void* structuredErrorContext)
    : m_oldCachedResourceLoader(currentCachedResourceLoader())
    , m_oldGenericErrorFunc(xmlGenericError)
    , m_oldStructuredErrorFunc(xmlStructuredError)
    , m_oldGenericErrorContext(xmlGenericErrorContext)
    , m_oldStructuredErrorContext(xmlStructuredErrorContext)
{
    initializeXMLParser();
    m_oldEntityLoader = xmlGetExternalEntityLoader();
    currentCachedResourceLoader() = cachedResourceLoader;
    xmlSetExternalEntityLoader(WebCore::externalEntityLoader);
    if (genericErrorFunc)
        xmlSetGenericErrorFunc(genericErrorContext, genericErrorFunc);
    if (structuredErrorFunc)
        xmlSetStructuredErrorFunc(structuredErrorContext ?: genericErrorContext, structuredErrorFunc);
}
#endif

XMLDocumentParserScope::~XMLDocumentParserScope()
{
    currentCachedResourceLoader() = m_oldCachedResourceLoader;
    xmlSetExternalEntityLoader(m_oldEntityLoader);
#if ENABLE(XSLT)
    xmlSetGenericErrorFunc(m_oldGenericErrorContext, m_oldGenericErrorFunc);
    xmlSetStructuredErrorFunc(m_oldStructuredErrorContext, m_oldStructuredErrorFunc);
#endif
}

}
