/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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

#if ENABLE(XSLT)

#include "CachedResourceHandle.h"
#include "CachedStyleSheetClient.h"
#include "XSLStyleSheet.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class CachedXSLStyleSheet;

class XSLImportRule : private CachedStyleSheetClient {
    WTF_MAKE_TZONE_ALLOCATED(XSLImportRule);
public:
    XSLImportRule(XSLStyleSheet& parentSheet, const String& href);
    virtual ~XSLImportRule();
    
    const String& href() const { return m_strHref; }
    XSLStyleSheet* styleSheet() const { return m_styleSheet.get(); }

    XSLStyleSheet* parentStyleSheet() const { return m_parentStyleSheet.get(); }
    void setParentStyleSheet(XSLStyleSheet* styleSheet) { m_parentStyleSheet = styleSheet; }

    bool isLoading();
    void loadSheet();
    
private:
    void setXSLStyleSheet(const String& href, const URL& baseURL, const String& sheet) override;
    
    WeakPtr<XSLStyleSheet> m_parentStyleSheet;
    String m_strHref;
    RefPtr<XSLStyleSheet> m_styleSheet;
    CachedResourceHandle<CachedXSLStyleSheet> m_cachedSheet;
    bool m_loading { false };
};

} // namespace WebCore

#endif // ENABLE(XSLT)
