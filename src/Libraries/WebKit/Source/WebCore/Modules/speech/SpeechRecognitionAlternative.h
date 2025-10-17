/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 18, 2022.
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

#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class SpeechRecognitionAlternative final : public RefCounted<SpeechRecognitionAlternative> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SpeechRecognitionAlternative);
public:
    static Ref<SpeechRecognitionAlternative> create(String&& transcript, double confidence);

    const String& transcript() const { return m_transcript; }
    double confidence() const { return m_confidence; }

private:
    SpeechRecognitionAlternative(String&& transcript, double confidence);

    String m_transcript;
    double m_confidence;
};

} // namespace WebCore
