/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 4, 2022.
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
#include "SpeechRecognitionResult.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SpeechRecognitionResult);

Ref<SpeechRecognitionResult> SpeechRecognitionResult::create(Vector<Ref<SpeechRecognitionAlternative>>&& alternatives, bool isFinal)
{
    return adoptRef(*new SpeechRecognitionResult(WTFMove(alternatives), isFinal));
}

SpeechRecognitionResult::SpeechRecognitionResult(Vector<Ref<SpeechRecognitionAlternative>>&& alternatives, bool isFinal)
    : m_alternatives(WTFMove(alternatives))
    , m_isFinal(isFinal)
{
}

SpeechRecognitionAlternative* SpeechRecognitionResult::item(uint64_t index) const
{
    return (index < m_alternatives.size()) ? m_alternatives[index].ptr() : nullptr;
}

} // namespace WebCore
