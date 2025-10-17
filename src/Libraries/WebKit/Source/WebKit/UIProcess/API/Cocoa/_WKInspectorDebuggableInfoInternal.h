/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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
#import "_WKInspectorDebuggableInfo.h"

#import "APIDebuggableInfo.h"
#import "WKObject.h"

namespace WebKit {

template<> struct WrapperTraits<API::DebuggableInfo> {
    using WrapperClass = _WKInspectorDebuggableInfo;
};

}

@interface _WKInspectorDebuggableInfo () <WKObject> {
@package
    API::ObjectStorage<API::DebuggableInfo> _debuggableInfo;
}
@end

inline Inspector::DebuggableType fromWKInspectorDebuggableType(_WKInspectorDebuggableType debuggableType)
{
    switch (debuggableType) {
    case _WKInspectorDebuggableTypeITML:
        return Inspector::DebuggableType::ITML;
    case _WKInspectorDebuggableTypeJavaScript:
        return Inspector::DebuggableType::JavaScript;
    case _WKInspectorDebuggableTypePage:
        return Inspector::DebuggableType::Page;
    case _WKInspectorDebuggableTypeServiceWorker:
        return Inspector::DebuggableType::ServiceWorker;
    case _WKInspectorDebuggableTypeWebPage:
        return Inspector::DebuggableType::WebPage;
    }

    ASSERT_NOT_REACHED();
    return Inspector::DebuggableType::JavaScript;
}

inline _WKInspectorDebuggableType toWKInspectorDebuggableType(Inspector::DebuggableType debuggableType)
{
    switch (debuggableType) {
    case Inspector::DebuggableType::ITML:
        return _WKInspectorDebuggableTypeITML;
    case Inspector::DebuggableType::JavaScript:
        return _WKInspectorDebuggableTypeJavaScript;
    case Inspector::DebuggableType::Page:
        return _WKInspectorDebuggableTypePage;
    case Inspector::DebuggableType::ServiceWorker:
        return _WKInspectorDebuggableTypeServiceWorker;
    case Inspector::DebuggableType::WebPage:
        return _WKInspectorDebuggableTypeWebPage;
    }

    ASSERT_NOT_REACHED();
    return _WKInspectorDebuggableTypeJavaScript;
}
