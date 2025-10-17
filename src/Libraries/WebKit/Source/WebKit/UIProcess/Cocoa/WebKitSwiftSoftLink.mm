/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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
#include <wtf/FileSystem.h>
#include <wtf/SoftLinking.h>

namespace WebKit {

extern void* WebKitSwiftLibrary(bool isOptional = false);
void* WebKitSwiftLibrary(bool isOptional)
{
    static void* library;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        // Start by searching for the library in DYLD_LIBRARY_PATH:
        if ((library = dlopen("libWebKitSwift.dylib", RTLD_NOW)))
            return;

        // Then search in the Frameworks/ directory of the currently loaded version of WebKit.framework:
        Dl_info info { };
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        if (dladdr((const void*)&WebKitSwiftLibrary, &info) && strlen(info.dli_fname)) {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
            auto dliPath = String::fromUTF8(info.dli_fname);
            if (dliPath.isNull())
                return;
            auto webkitFrameworkDirectory = WTF::FileSystemImpl::parentPath(dliPath);
            auto dylibPath = WTF::FileSystemImpl::pathByAppendingComponent(webkitFrameworkDirectory, "Frameworks/libWebKitSwift.dylib"_s);
            if ((library = dlopen(dylibPath.utf8().data(), RTLD_NOW)))
                return;
        }

        if (!isOptional)
            RELEASE_ASSERT_WITH_MESSAGE(library, "%s", dlerror());
    });
    return library;
}

}

SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKGroupSessionObserver)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKSLinearMediaContentMetadata)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKSLinearMediaPlayer)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKSLinearMediaTimeRange)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKSLinearMediaTrack)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKSPreviewWindowController)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKSRKEntity)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKSTextAnimationManager)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKIntelligenceReplacementTextEffectCoordinator)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKIntelligenceSmartReplyTextEffectCoordinator)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKTextExtractionContainerItem)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKTextExtractionEditable)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKTextExtractionLink)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKTextExtractionTextItem)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKTextExtractionScrollableItem)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKTextExtractionImageItem)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, WebKitSwift, WKSLinearMediaSpatialVideoMetadata)
