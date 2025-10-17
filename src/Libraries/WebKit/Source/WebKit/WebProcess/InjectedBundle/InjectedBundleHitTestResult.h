/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
#ifndef InjectedBundleHitTestResult_h
#define InjectedBundleHitTestResult_h

#include "APIObject.h"
#include "InjectedBundleHitTestResultMediaType.h"
#include <WebCore/HitTestResult.h>
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>

namespace WebKit {

class InjectedBundleNodeHandle;
class WebFrame;
class WebImage;

class InjectedBundleHitTestResult : public API::ObjectImpl<API::Object::Type::BundleHitTestResult> {
public:
    static Ref<InjectedBundleHitTestResult> create(const WebCore::HitTestResult&);

    const WebCore::HitTestResult& coreHitTestResult() const { return m_hitTestResult; }

    RefPtr<InjectedBundleNodeHandle> nodeHandle() const;
    RefPtr<InjectedBundleNodeHandle> urlElementHandle() const;
    RefPtr<WebFrame> frame() const;
    RefPtr<WebFrame> targetFrame() const;

    String absoluteImageURL() const;
    String absolutePDFURL() const;
    String absoluteLinkURL() const;
    String absoluteMediaURL() const;
    bool mediaIsInFullscreen() const;
    bool mediaHasAudio() const;
    bool isDownloadableMedia() const;
    BundleHitTestResultMediaType mediaType() const;

    String linkLabel() const;
    String linkTitle() const;
    String linkSuggestedFilename() const;
    
    WebCore::IntRect imageRect() const;
    RefPtr<WebImage> image() const;
    
    bool isSelected() const;

private:
    explicit InjectedBundleHitTestResult(const WebCore::HitTestResult& hitTestResult)
        : m_hitTestResult(hitTestResult)
    {
    }

    WebCore::HitTestResult m_hitTestResult;
};

} // namespace WebKit

#endif // InjectedBundleHitTestResult_h
