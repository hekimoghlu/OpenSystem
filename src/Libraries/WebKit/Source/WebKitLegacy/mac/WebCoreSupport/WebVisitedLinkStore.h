/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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
#ifndef WebVisitedLinkStore_h
#define WebVisitedLinkStore_h

#import <WebCore/SharedStringHash.h>
#import <WebCore/VisitedLinkStore.h>
#import <wtf/CheckedRef.h>
#import <wtf/Ref.h>

class WebVisitedLinkStore final : public WebCore::VisitedLinkStore, public CanMakeWeakPtr<WebVisitedLinkStore> {
public:
    static Ref<WebVisitedLinkStore> create();
    virtual ~WebVisitedLinkStore();

    static void setShouldTrackVisitedLinks(bool);
    static void removeAllVisitedLinks();

    void addVisitedLink(NSString *urlString);
    void removeVisitedLink(NSString *urlString);

private:
    WebVisitedLinkStore();

    bool isLinkVisited(WebCore::Page&, WebCore::SharedStringHash, const URL& baseURL, const AtomString& attributeURL) override;
    void addVisitedLink(WebCore::Page&, WebCore::SharedStringHash) override;

    void populateVisitedLinksIfNeeded(WebCore::Page&);
    void addVisitedLinkHash(WebCore::SharedStringHash);
    void removeVisitedLinkHashes();

    HashSet<WebCore::SharedStringHash, WebCore::SharedStringHashHash> m_visitedLinkHashes;
    bool m_visitedLinksPopulated;
};

#endif // WebVisitedLinkStore_h
