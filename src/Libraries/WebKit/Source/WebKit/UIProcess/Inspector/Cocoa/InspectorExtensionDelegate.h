/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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

#if ENABLE(INSPECTOR_EXTENSIONS)

#import "APIInspectorExtensionClient.h"
#import "WKFoundation.h"
#import <WebCore/FrameIdentifier.h>
#import <wtf/CheckedPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakObjCPtr.h>

@class _WKInspectorExtension;
@protocol _WKInspectorExtensionDelegate;

namespace WebKit {

class InspectorExtensionDelegate : public CanMakeCheckedPtr<InspectorExtensionDelegate> {
    WTF_MAKE_TZONE_ALLOCATED(InspectorExtensionDelegate);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(InspectorExtensionDelegate);
public:
    InspectorExtensionDelegate(_WKInspectorExtension *, id <_WKInspectorExtensionDelegate>);
    ~InspectorExtensionDelegate();

    RetainPtr<id <_WKInspectorExtensionDelegate>> delegate();
    void setDelegate(id <_WKInspectorExtensionDelegate>);

private:
    class InspectorExtensionClient final : public API::InspectorExtensionClient {
        WTF_MAKE_TZONE_ALLOCATED(InspectorExtensionClient);
    public:
        explicit InspectorExtensionClient(InspectorExtensionDelegate&);
        ~InspectorExtensionClient();

    private:
        // API::InspectorExtensionClient
        void didShowExtensionTab(const Inspector::ExtensionTabID&, WebCore::FrameIdentifier) override;
        void didHideExtensionTab(const Inspector::ExtensionTabID&) override;
        void didNavigateExtensionTab(const Inspector::ExtensionTabID&, const URL&) override;
        void inspectedPageDidNavigate(const URL&) override;

        CheckedRef<InspectorExtensionDelegate> m_inspectorExtensionDelegate;
    };

    WeakObjCPtr<_WKInspectorExtension> m_inspectorExtension;
    WeakObjCPtr<id <_WKInspectorExtensionDelegate>> m_delegate;

    struct {
        bool inspectorExtensionDidShowTabWithIdentifier : 1;
        bool inspectorExtensionDidHideTabWithIdentifier : 1;
        bool inspectorExtensionDidNavigateTabWithIdentifier : 1;
        bool inspectorExtensionInspectedPageDidNavigate : 1;
    } m_delegateMethods;
};

} // namespace WebKit

#endif // ENABLE(INSPECTOR_EXTENSIONS)
