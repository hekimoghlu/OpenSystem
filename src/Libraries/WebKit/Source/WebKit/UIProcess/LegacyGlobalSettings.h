/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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

#include "CacheModel.h"
#include <wtf/Forward.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class LegacyGlobalSettings {
    WTF_MAKE_TZONE_ALLOCATED(LegacyGlobalSettings);
public:
    static LegacyGlobalSettings& singleton();

    void setCacheModel(CacheModel);
    CacheModel cacheModel() const { return m_cacheModel; }

    const HashSet<String>& schemesToRegisterAsSecure() { return m_schemesToRegisterAsSecure; }
    void registerURLSchemeAsSecure(const String& scheme) { m_schemesToRegisterAsSecure.add(scheme); }

    const HashSet<String>& schemesToRegisterAsBypassingContentSecurityPolicy() { return m_schemesToRegisterAsBypassingContentSecurityPolicy; }
    void registerURLSchemeAsBypassingContentSecurityPolicy(const String& scheme) { m_schemesToRegisterAsBypassingContentSecurityPolicy.add(scheme); }

    const HashSet<String>& schemesToRegisterAsLocal() { return m_schemesToRegisterAsLocal; }
    void registerURLSchemeAsLocal(const String& scheme) { m_schemesToRegisterAsLocal.add(scheme); }

    const HashSet<String>& schemesToRegisterAsNoAccess() { return m_schemesToRegisterAsNoAccess; }
    void registerURLSchemeAsNoAccess(const String& scheme) { m_schemesToRegisterAsNoAccess.add(scheme); }

    const HashSet<String>& hostnamesToRegisterAsLocal() const { return m_hostnamesToRegisterAsLocal; }
    void registerHostnameAsLocal(const String& hostname) { m_hostnamesToRegisterAsLocal.add(hostname); }

private:
    friend class NeverDestroyed<LegacyGlobalSettings>;
    LegacyGlobalSettings();
    
    CacheModel m_cacheModel { CacheModel::PrimaryWebBrowser };
    HashSet<String> m_schemesToRegisterAsSecure;
    HashSet<String> m_schemesToRegisterAsBypassingContentSecurityPolicy;
    HashSet<String> m_schemesToRegisterAsLocal;
    HashSet<String> m_schemesToRegisterAsNoAccess;
    HashSet<String> m_hostnamesToRegisterAsLocal;
};

bool experimentalFeatureEnabled(const String& key, bool defaultValue = false);

} // namespace WebKit
