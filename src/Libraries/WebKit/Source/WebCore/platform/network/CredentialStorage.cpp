/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 22, 2025.
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
#include "CredentialStorage.h"

#include "NetworkStorageSession.h"
#include <wtf/URL.h>
#include <wtf/text/MakeString.h>

#if PLATFORM(IOS_FAMILY)
#include "WebCoreThread.h"
#endif

namespace WebCore {

static String originStringFromURL(const URL& url)
{
    return makeString(url.protocol(), "://"_s, url.hostAndPort(), '/');
}

static String protectionSpaceMapKeyFromURL(const URL& url)
{
    ASSERT(url.isValid());

    // Remove the last path component that is not a directory to determine the subtree for which credentials will apply.
    // We keep a leading slash, but remove a trailing one.
    String directoryURL = url.string().left(url.pathEnd());
    unsigned directoryURLPathStart = url.pathStart();
    ASSERT(directoryURL[directoryURLPathStart] == '/');
    if (directoryURL.length() > directoryURLPathStart + 1) {
        size_t index = directoryURL.reverseFind('/');
        ASSERT(index != notFound);
        directoryURL = directoryURL.left((index != directoryURLPathStart) ? index : directoryURLPathStart + 1);
    }

    return directoryURL;
}

void CredentialStorage::set(const String& partitionName, const Credential& credential, const ProtectionSpace& protectionSpace, const URL& url)
{
    ASSERT(protectionSpace.isProxy() || protectionSpace.authenticationScheme() == ProtectionSpace::AuthenticationScheme::ClientCertificateRequested || url.protocolIsInHTTPFamily());
    ASSERT(protectionSpace.isProxy() || protectionSpace.authenticationScheme() == ProtectionSpace::AuthenticationScheme::ClientCertificateRequested || url.isValid());

    m_protectionSpaceToCredentialMap.set(std::make_pair(partitionName, protectionSpace), credential);

    if (!protectionSpace.isProxy() && protectionSpace.authenticationScheme() != ProtectionSpace::AuthenticationScheme::ClientCertificateRequested) {
        m_originsWithCredentials.add(originStringFromURL(url));

        auto scheme = protectionSpace.authenticationScheme();
        if (scheme == ProtectionSpace::AuthenticationScheme::HTTPBasic || scheme == ProtectionSpace::AuthenticationScheme::Default) {
            // The map can contain both a path and its subpath - while redundant, this makes lookups faster.
            m_pathToDefaultProtectionSpaceMap.set(protectionSpaceMapKeyFromURL(url), protectionSpace);
        }
    }
}

Credential CredentialStorage::get(const String& partitionName, const ProtectionSpace& protectionSpace)
{
    return m_protectionSpaceToCredentialMap.get(std::make_pair(partitionName, protectionSpace));
}

void CredentialStorage::remove(const String& partitionName, const ProtectionSpace& protectionSpace)
{
    m_protectionSpaceToCredentialMap.remove(std::make_pair(partitionName, protectionSpace));
}

void CredentialStorage::removeCredentialsWithOrigin(const SecurityOriginData& origin)
{
    Vector<std::pair<String, ProtectionSpace>> keysToRemove;
    for (auto& keyValuePair : m_protectionSpaceToCredentialMap) {
        auto& protectionSpace = keyValuePair.key.second;
        if (protectionSpace.host() == origin.host()
            && ((origin.port() && protectionSpace.port() == *origin.port())
                || (!origin.port() && protectionSpace.port() == 80))
            && ((protectionSpace.serverType() == ProtectionSpace::ServerType::HTTP && origin.protocol() == "http"_s)
                || (protectionSpace.serverType() == ProtectionSpace::ServerType::HTTPS && origin.protocol() == "https"_s)))
            keysToRemove.append(keyValuePair.key);
    }
    for (auto& key : keysToRemove)
        remove(key.first, key.second);
}

HashSet<SecurityOriginData> CredentialStorage::originsWithCredentials() const
{
    HashSet<SecurityOriginData> origins;
    for (auto& keyValuePair : m_protectionSpaceToCredentialMap) {
        auto& protectionSpace = keyValuePair.key.second;
        if (protectionSpace.isProxy())
            continue;
        String protocol;
        switch (protectionSpace.serverType()) {
        case ProtectionSpace::ServerType::HTTP:
            protocol = "http"_s;
            break;
        case ProtectionSpace::ServerType::HTTPS:
            protocol = "https"_s;
            break;
        case ProtectionSpace::ServerType::FTP:
            protocol = "ftp"_s;
            break;
        case ProtectionSpace::ServerType::FTPS:
            protocol = "ftps"_s;
            break;
        default:
            ASSERT_NOT_REACHED();
            continue;
        }

        SecurityOriginData origin { protocol, protectionSpace.host(), static_cast<uint16_t>(protectionSpace.port())};
        origins.add(WTFMove(origin));
    }
    return origins;
}

HashMap<String, ProtectionSpace>::iterator CredentialStorage::findDefaultProtectionSpaceForURL(const URL& url)
{
    ASSERT(url.protocolIsInHTTPFamily());
    ASSERT(url.isValid());

    // Don't spend time iterating the path for origins that don't have any credentials.
    if (!m_originsWithCredentials.contains(originStringFromURL(url)))
        return m_pathToDefaultProtectionSpaceMap.end();

    String directoryURL = protectionSpaceMapKeyFromURL(url);
    unsigned directoryURLPathStart = url.pathStart();
    while (true) {
        PathToDefaultProtectionSpaceMap::iterator iter = m_pathToDefaultProtectionSpaceMap.find(directoryURL);
        if (iter != m_pathToDefaultProtectionSpaceMap.end())
            return iter;

        if (directoryURL.length() == directoryURLPathStart + 1)  // path is "/" already, cannot shorten it any more
            return m_pathToDefaultProtectionSpaceMap.end();

        size_t index = directoryURL.reverseFind('/', directoryURL.length() - 2);
        ASSERT(index != notFound);
        directoryURL = directoryURL.left((index == directoryURLPathStart) ? index + 1 : index);
        ASSERT(directoryURL.length() > directoryURLPathStart);
    }
}

bool CredentialStorage::set(const String& partitionName, const Credential& credential, const URL& url)
{
    ASSERT(url.protocolIsInHTTPFamily());
    ASSERT(url.isValid());
    PathToDefaultProtectionSpaceMap::iterator iter = findDefaultProtectionSpaceForURL(url);
    if (iter == m_pathToDefaultProtectionSpaceMap.end())
        return false;
    ASSERT(m_originsWithCredentials.contains(originStringFromURL(url)));
    m_protectionSpaceToCredentialMap.set(std::make_pair(partitionName, iter->value), credential);
    return true;
}

Credential CredentialStorage::get(const String& partitionName, const URL& url)
{
    PathToDefaultProtectionSpaceMap::iterator iter = findDefaultProtectionSpaceForURL(url);
    if (iter == m_pathToDefaultProtectionSpaceMap.end())
        return Credential();
    return m_protectionSpaceToCredentialMap.get(std::make_pair(partitionName, iter->value));
}

void CredentialStorage::clearCredentials()
{
    m_protectionSpaceToCredentialMap.clear();
    m_originsWithCredentials.clear();
    m_pathToDefaultProtectionSpaceMap.clear();
}

} // namespace WebCore
