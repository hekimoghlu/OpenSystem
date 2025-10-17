/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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

#if ENABLE(CONTEXT_MENUS)

#include <wtf/Forward.h>

namespace WebCore {

class IntPoint;
class IntRect;
class LocalFrame;

#if HAVE(TRANSLATION_UI_SERVICES)
struct TranslationContextMenuInfo;
#endif

class ContextMenuClient {
public:
    virtual ~ContextMenuClient() = default;
    
    virtual void downloadURL(const URL&) = 0;
    virtual void searchWithGoogle(const LocalFrame*) = 0;
    virtual void lookUpInDictionary(LocalFrame*) = 0;
    virtual bool isSpeaking() const = 0;
    virtual void speak(const String&) = 0;
    virtual void stopSpeaking() = 0;

#if ENABLE(IMAGE_ANALYSIS)
    virtual bool supportsLookUpInImages() = 0;
#endif

#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)
    virtual bool supportsCopySubject() = 0;
#endif

#if HAVE(TRANSLATION_UI_SERVICES)
    virtual void handleTranslation(const TranslationContextMenuInfo&) = 0;
#endif

#if PLATFORM(GTK)
    virtual void insertEmoji(LocalFrame&) = 0;
#endif

#if USE(ACCESSIBILITY_CONTEXT_MENUS)
    virtual void showContextMenu() = 0;
#endif
};

} // namespace WebCore

#endif // ENABLE(CONTEXT_MENUS)
