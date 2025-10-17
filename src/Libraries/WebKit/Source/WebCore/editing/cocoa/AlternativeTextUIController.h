/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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
#import "AlternativeTextContextController.h"
#import <wtf/TZoneMalloc.h>

@class NSView;

namespace WebCore {

class FloatRect;

class AlternativeTextUIController {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(AlternativeTextUIController, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT std::optional<DictationContext> addAlternatives(PlatformTextAlternatives *);
    WEBCORE_EXPORT void replaceAlternatives(PlatformTextAlternatives *, DictationContext);
    WEBCORE_EXPORT void removeAlternatives(DictationContext);
    WEBCORE_EXPORT void clear();

    WEBCORE_EXPORT PlatformTextAlternatives *alternativesForContext(DictationContext);

#if USE(APPKIT)
    using AcceptanceHandler = void (^)(NSString *);
    WEBCORE_EXPORT void showAlternatives(NSView *, const FloatRect& boundingBoxOfPrimaryString, DictationContext, AcceptanceHandler);
#endif

private:
#if USE(APPKIT)
    void handleAcceptedAlternative(NSString *, DictationContext, PlatformTextAlternatives *);
    void dismissAlternatives();
#endif

    AlternativeTextContextController m_contextController;
#if USE(APPKIT)
    RetainPtr<NSView> m_view;
#endif
};

} // namespace WebCore
