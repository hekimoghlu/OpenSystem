/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 21, 2024.
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

#include "SpeechRecognitionConnectionClientIdentifier.h"
#include <wtf/Identified.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class SpeechRecognitionConnectionClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::SpeechRecognitionConnectionClient> : std::true_type { };
}

namespace WebCore {

struct SpeechRecognitionError;
struct SpeechRecognitionResultData;

class SpeechRecognitionConnectionClient : public Identified<SpeechRecognitionConnectionClientIdentifier>, public CanMakeWeakPtr<SpeechRecognitionConnectionClient> {
public:
    SpeechRecognitionConnectionClient() = default;

    virtual ~SpeechRecognitionConnectionClient() { }

    virtual void didStart() = 0;
    virtual void didStartCapturingAudio() = 0;
    virtual void didStartCapturingSound() = 0;
    virtual void didStartCapturingSpeech() = 0;
    virtual void didStopCapturingSpeech() = 0;
    virtual void didStopCapturingSound() = 0;
    virtual void didStopCapturingAudio() = 0;
    virtual void didFindNoMatch() = 0;
    virtual void didReceiveResult(Vector<SpeechRecognitionResultData>&& resultDatas) = 0;
    virtual void didError(const SpeechRecognitionError&) = 0;
    virtual void didEnd() = 0;
};

} // namespace WebCore
