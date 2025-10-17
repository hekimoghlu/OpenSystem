/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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
#import <WebCore/DocumentLoader.h>
#import <WebCore/ResourceLoaderIdentifier.h>
#import <wtf/RetainPtr.h>
#import <wtf/HashSet.h>

@class WebDataSource;
@class WebView;

namespace WebCore {
class ResourceRequest;
}

class WebDocumentLoaderMac : public WebCore::DocumentLoader {
public:
    static Ref<WebDocumentLoaderMac> create(const WebCore::ResourceRequest& request, const WebCore::SubstituteData& data)
    {
        return adoptRef(*new WebDocumentLoaderMac(request, data));
    }

    void setDataSource(WebDataSource *, WebView*);
    void detachDataSource();
    WebDataSource *dataSource() const;

    void increaseLoadCount(WebCore::ResourceLoaderIdentifier);
    void decreaseLoadCount(WebCore::ResourceLoaderIdentifier);

private:
    WebDocumentLoaderMac(const WebCore::ResourceRequest&, const WebCore::SubstituteData&);

    virtual void attachToFrame();
    virtual void detachFromFrame(WebCore::LoadWillContinueInAnotherProcess);

    void retainDataSource();
    void releaseDataSource();

    WebDataSource *m_dataSource;
    bool m_isDataSourceRetained;
    RetainPtr<id> m_resourceLoadDelegate;
    RetainPtr<id> m_downloadDelegate;
    HashSet<WebCore::ResourceLoaderIdentifier> m_loadingResources;
};
