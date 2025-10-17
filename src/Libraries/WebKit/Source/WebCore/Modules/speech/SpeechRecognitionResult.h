/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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

#include "SpeechRecognitionAlternative.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SpeechRecognitionResult final : public RefCounted<SpeechRecognitionResult> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SpeechRecognitionResult);
public:
    static Ref<SpeechRecognitionResult> create(Vector<Ref<SpeechRecognitionAlternative>>&&, bool isFinal);

    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }
    uint64_t length() const { return m_alternatives.size(); }
    bool isFinal() const { return m_isFinal; }
    SpeechRecognitionAlternative* item(uint64_t index) const;

private:
    SpeechRecognitionResult(Vector<Ref<SpeechRecognitionAlternative>>&&, bool isFinal);

    Vector<Ref<SpeechRecognitionAlternative>> m_alternatives;
    bool m_isFinal { false };
};

}
